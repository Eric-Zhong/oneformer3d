# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load Scannet scenes with vertices and ground truth labels for semantic and
instance segmentations."""
import argparse  # 导入argparse模块，用于处理命令行参数
import inspect  # 导入inspect模块，用于获取当前文件信息
import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于处理文件和目录

import numpy as np  # 导入numpy库，用于数值计算
import scannet_utils  # 导入自定义的scannet_utils模块

# 获取当前文件的目录
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))

def read_aggregation(filename):
    """读取聚合文件，返回对象ID到分段的映射和标签到分段的映射"""
    assert os.path.isfile(filename)  # 确保文件存在
    object_id_to_segs = {}  # 创建对象ID到分段的映射
    label_to_segs = {}  # 创建标签到分段的映射
    with open(filename) as f:  # 打开文件
        data = json.load(f)  # 读取JSON数据
        num_objects = len(data['segGroups'])  # 获取对象数量
        for i in range(num_objects):  # 遍历每个对象
            object_id = data['segGroups'][i][
                'objectId'] + 1  # 实例ID应为1索引
            label = data['segGroups'][i]['label']  # 获取标签
            segs = data['segGroups'][i]['segments']  # 获取分段
            object_id_to_segs[object_id] = segs  # 更新对象ID到分段的映射
            if label in label_to_segs:  # 如果标签已存在
                label_to_segs[label].extend(segs)  # 扩展分段
            else:
                label_to_segs[label] = segs  # 创建新的标签映射
    return object_id_to_segs, label_to_segs  # 返回映射

def read_segmentation(filename):
    """读取分段文件，返回分段到顶点的映射和顶点数量"""
    assert os.path.isfile(filename)  # 确保文件存在
    seg_to_verts = {}  # 创建分段到顶点的映射
    with open(filename) as f:  # 打开文件
        data = json.load(f)  # 读取JSON数据
        num_verts = len(data['segIndices'])  # 获取顶点数量
        for i in range(num_verts):  # 遍历每个顶点
            seg_id = data['segIndices'][i]  # 获取分段ID
            if seg_id in seg_to_verts:  # 如果分段ID已存在
                seg_to_verts[seg_id].append(i)  # 添加顶点索引
            else:
                seg_to_verts[seg_id] = [i]  # 创建新的分段映射
    return seg_to_verts, num_verts  # 返回映射和顶点数量

def extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id,
                 instance_ids):
    """从网格顶点提取边界框"""
    num_instances = len(np.unique(list(object_id_to_segs.keys())))  # 获取实例数量
    instance_bboxes = np.zeros((num_instances, 7))  # 初始化边界框数组
    for obj_id in object_id_to_segs:  # 遍历每个对象ID
        label_id = object_id_to_label_id[obj_id]  # 获取标签ID
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]  # 获取对象点云
        if len(obj_pc) == 0:  # 如果点云为空
            continue  # 跳过
        xyz_min = np.min(obj_pc, axis=0)  # 获取最小坐标
        xyz_max = np.max(obj_pc, axis=0)  # 获取最大坐标
        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min,
                               np.array([label_id])])  # 计算边界框
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox  # 更新边界框数组
    return instance_bboxes  # 返回边界框

def export(mesh_file,
           agg_file,
           seg_file,
           meta_file,
           label_map_file,
           output_file=None,
           test_mode=False,
           scannet200=False):
    """导出原始文件为顶点、实例标签、语义标签和边界框文件"""
    if scannet200:  # 如果使用scannet200
        label_map = scannet_utils.read_label_mapping(
            label_map_file, label_from='raw_category', label_to='id')  # 读取标签映射
    else:
        label_map = scannet_utils.read_label_mapping(
            label_map_file, label_from='raw_category', label_to='nyu40id')  # 读取标签映射

    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)  # 读取网格顶点

    # 加载场景轴对齐矩阵
    lines = open(meta_file).readlines()  # 读取元数据文件
    # 测试集数据没有align_matrix
    axis_align_matrix = np.eye(4)  # 初始化为单位矩阵
    for line in lines:  # 遍历每一行
        if 'axisAlignment' in line:  # 如果包含轴对齐信息
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]  # 解析轴对齐矩阵
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))  # 转换为4x4矩阵

    # 对网格顶点进行全局对齐
    pts = np.ones((mesh_vertices.shape[0], 4))  # 创建齐次坐标
    pts[:, 0:3] = mesh_vertices[:, 0:3]  # 复制顶点坐标
    pts = np.dot(pts, axis_align_matrix.transpose())  # 进行矩阵乘法
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]],
                                           axis=1)  # 合并对齐后的顶点

    # 加载语义和实例标签
    if not test_mode:  # 如果不是测试模式
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)  # 读取聚合数据
        seg_to_verts, num_verts = read_segmentation(seg_file)  # 读取分段数据
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 初始化标签数组
        object_id_to_label_id = {}  # 创建对象ID到标签ID的映射
        for label, segs in label_to_segs.items():  # 遍历标签到分段的映射
            label_id = label_map[label]  # 获取标签ID
            for seg in segs:  # 遍历每个分段
                verts = seg_to_verts[seg]  # 获取顶点索引
                label_ids[verts] = label_id  # 更新标签数组
        instance_ids = np.zeros(
            shape=(num_verts), dtype=np.uint32)  # 初始化实例ID数组
        for object_id, segs in object_id_to_segs.items():  # 遍历对象ID到分段的映射
            for seg in segs:  # 遍历每个分段
                verts = seg_to_verts[seg]  # 获取顶点索引
                instance_ids[verts] = object_id  # 更新实例ID数组
                if object_id not in object_id_to_label_id:  # 如果对象ID不在标签映射中
                    object_id_to_label_id[object_id] = label_ids[verts][0]  # 更新标签ID映射
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)  # 提取未对齐的边界框
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)  # 提取对齐的边界框
    else:  # 如果是测试模式
        label_ids = None  # 标签ID为None
        instance_ids = None  # 实例ID为None
        unaligned_bboxes = None  # 未对齐的边界框为None
        aligned_bboxes = None  # 对齐的边界框为None
        object_id_to_label_id = None  # 对象ID到标签ID的映射为None

    if output_file is not None:  # 如果指定了输出文件
        np.save(output_file + '_vert.npy', mesh_vertices)  # 保存顶点数据
        if not test_mode:  # 如果不是测试模式
            np.save(output_file + '_sem_label.npy', label_ids)  # 保存语义标签
            np.save(output_file + '_ins_label.npy', instance_ids)  # 保存实例标签
            np.save(output_file + '_unaligned_bbox.npy', unaligned_bboxes)  # 保存未对齐的边界框
            np.save(output_file + '_aligned_bbox.npy', aligned_bboxes)  # 保存对齐的边界框
            np.save(output_file + '_axis_align_matrix.npy', axis_align_matrix)  # 保存轴对齐矩阵

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix  # 返回结果

def main():
    """主函数，解析命令行参数并调用导出函数"""
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        '--scan_path',
        required=True,
        help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')  # 添加扫描路径参数
    )
    parser.add_argument('--output_file', required=True, help='output file')  # 添加输出文件参数
    parser.add_argument(
        '--label_map_file',
        required=True,
        help='path to scannetv2-labels.combined.tsv')  # 添加标签映射文件参数
    parser.add_argument(
        '--scannet200',
        action='store_true',
        help='Use it for scannet200 mapping')  # 添加scannet200标志参数

    opt = parser.parse_args()  # 解析参数

    scan_name = os.path.split(opt.scan_path)[-1]  # 获取扫描名称
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')  # 构建网格文件路径
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')  # 构建聚合文件路径
    seg_file = os.path.join(opt.scan_path,
                            scan_name + '_vh_clean_2.0.010000.segs.json')  # 构建分段文件路径
    meta_file = os.path.join(
        opt.scan_path, scan_name +
        '.txt')  # 构建元数据文件路径
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file,
           opt.output_file, scannet200=opt.scannet200)  # 调用导出函数

if __name__ == '__main__':
    main()  # 运行主函数