# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""
import argparse  # 导入命令行参数解析库
import datetime  # 导入日期和时间库
import os  # 导入操作系统接口库
from os import path as osp  # 从os库导入路径处理功能并重命名为osp

import torch  # 导入PyTorch库
import segmentator  # 导入分割器库
import open3d as o3d  # 导入Open3D库用于3D数据处理
import numpy as np  # 导入NumPy库用于数组处理
from load_scannet_data import export  # 从load_scannet_data模块导入export函数

DONOTCARE_CLASS_IDS = np.array([])  # 定义不关心的类ID为空数组

SCANNET_OBJ_CLASS_IDS = np.array(  # 定义Scannet对象类ID数组
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

SCANNET200_OBJ_CLASS_IDS = np.array([  # 定义Scannet200对象类ID数组
    2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191])

def export_one_scan(scan_name,  # 定义导出单个扫描的函数
                    output_filename_prefix,  # 输出文件名前缀
                    max_num_point,  # 最大点数
                    label_map_file,  # 标签映射文件
                    scannet_dir,  # Scannet数据目录
                    test_mode=False,  # 测试模式标志
                    scannet200=False):  # 是否使用Scannet200标志

    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')  # 生成网格文件路径
    agg_file = osp.join(scannet_dir, scan_name, scan_name + '.aggregation.json')  # 生成聚合文件路径
    seg_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')  # 生成分割文件路径
    
    # 包含训练集扫描的轴对齐信息
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')  # 生成元数据文件路径
    # 从export函数获取数据
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode, scannet200)

    if not test_mode:  # 如果不是测试模式
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))  # 创建掩码，过滤不关心的类
        mesh_vertices = mesh_vertices[mask, :]  # 应用掩码到网格顶点
        semantic_labels = semantic_labels[mask]  # 应用掩码到语义标签
        instance_labels = instance_labels[mask]  # 应用掩码到实例标签

        num_instances = len(np.unique(instance_labels))  # 计算实例数量
        print(f'Num of instances: {num_instances}')  # 打印实例数量
        if scannet200:  # 如果使用Scannet200
            OBJ_CLASS_IDS = SCANNET200_OBJ_CLASS_IDS  # 使用Scannet200类ID
        else:
            OBJ_CLASS_IDS = SCANNET_OBJ_CLASS_IDS  # 使用Scannet类ID

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)  # 创建边界框掩码
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]  # 应用掩码到未对齐边界框
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)  # 创建对齐边界框掩码
        aligned_bboxes = aligned_bboxes[bbox_mask, :]  # 应用掩码到对齐边界框
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]  # 确保未对齐和对齐边界框数量相同
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')  # 打印关心的实例数量

    if max_num_point is not None:  # 如果最大点数不为None
        max_num_point = int(max_num_point)  # 转换为整数
        N = mesh_vertices.shape[0]  # 获取网格顶点数量
        if N > max_num_point:  # 如果网格顶点数量大于最大点数
            choices = np.random.choice(N, max_num_point, replace=False)  # 随机选择点
            mesh_vertices = mesh_vertices[choices, :]  # 应用选择到网格顶点
            if not test_mode:  # 如果不是测试模式
                semantic_labels = semantic_labels[choices]  # 应用选择到语义标签
                instance_labels = instance_labels[choices]  # 应用选择到实例标签
    
    mesh = o3d.io.read_triangle_mesh(mesh_file)  # 读取三角网格
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))  # 转换顶点为PyTorch张量
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))  # 转换面为PyTorch张量
    superpoints = segmentator.segment_mesh(vertices, faces).numpy()  # 对网格进行分割并转换为NumPy数组
    
    np.save(f'{output_filename_prefix}_sp_label.npy', superpoints)  # 保存超点标签
    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)  # 保存网格顶点

    if not test_mode:  # 如果不是测试模式
        assert superpoints.shape == semantic_labels.shape  # 确保超点标签和语义标签形状相同
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)  # 保存语义标签
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)  # 保存实例标签
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy', unaligned_bboxes)  # 保存未对齐边界框
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)  # 保存对齐边界框
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy', axis_align_matrix)  # 保存轴对齐矩阵


def batch_export(max_num_point,  # 定义批量导出的函数
                 output_folder,  # 输出文件夹
                 scan_names_file,  # 扫描名称文件
                 label_map_file,  # 标签映射文件
                 scannet_dir,  # Scannet数据目录
                 test_mode=False,  # 测试模式标志
                 scannet200=False):  # 是否使用Scannet200标志
    if test_mode and not os.path.exists(scannet_dir):  # 如果是测试模式且Scannet目录不存在
        # 测试数据准备是可选的
        return  # 直接返回
    if not os.path.exists(output_folder):  # 如果输出文件夹不存在
        print(f'Creating new data folder: {output_folder}')  # 打印创建新文件夹信息
        os.mkdir(output_folder)  # 创建新文件夹

    scan_names = [line.rstrip() for line in open(scan_names_file)]  # 读取扫描名称文件并去除换行符
    for scan_name in scan_names:  # 遍历每个扫描名称
        print('-' * 20 + 'begin')  # 打印开始标志
        print(datetime.datetime.now())  # 打印当前时间
        print('-------------------- OK --------------------')
        print(scan_name)  # 打印当前扫描名称
        output_filename_prefix = osp.join(output_folder, scan_name)  # 生成输出文件名前缀
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):  # 如果输出文件已存在
            print('File already exists. skipping.')  # 打印跳过信息
            print('-' * 20 + 'done')  # 打印完成标志
            continue  # 跳过当前扫描名称
        try:
            export_one_scan( # 调用导出单个扫描函数
                scan_name, 
                output_filename_prefix, 
                max_num_point,  
                label_map_file, 
                scannet_dir, 
                test_mode, 
                scannet200
            )
        except Exception:  # 捕获异常
            print(f'Failed export scan: {scan_name}')  # 打印导出失败信息

        print('-' * 20 + 'done')  # 打印完成标志


def main():  # 定义主函数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(  # 添加最大点数参数
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')  # 参数说明
    parser.add_argument(  # 添加输出文件夹参数
        '--output_folder',
        default='./scannet_instance_data',
        help='output folder of the result.')  # 参数说明
    parser.add_argument(  # 添加训练Scannet目录参数
        '--train_scannet_dir', default='scans', help='scannet data directory.')  # 参数说明
    parser.add_argument(  # 添加测试Scannet目录参数
        '--test_scannet_dir',
        default='scans_test',
        help='scannet data directory.')  # 参数说明
    parser.add_argument(  # 添加标签映射文件参数
        '--label_map_file',
        default='meta_data/scannetv2-labels.combined.tsv',
        help='The path of label map file.')  # 参数说明
    parser.add_argument(  # 添加训练扫描名称文件参数
        '--train_scan_names_file',
        default='meta_data/scannet_train.txt',
        help='The path of the file that stores the scan names.')  # 参数说明
    parser.add_argument(  # 添加测试扫描名称文件参数
        '--test_scan_names_file',
        default='meta_data/scannetv2_test.txt',
        help='The path of the file that stores the scan names.')  # 参数说明
    parser.add_argument(  # 添加Scannet200标志参数
        '--scannet200',
        action='store_true',
        help='Use it for scannet200 mapping')  # 参数说明
    args = parser.parse_args()  # 解析参数
    batch_export(  # 调用批量导出函数
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        test_mode=False,
        scannet200=args.scannet200)
    batch_export(  # 调用批量导出函数
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        test_mode=True,
        scannet200=args.scannet200)


if __name__ == '__main__':  # 如果是主模块
    main()  # 调用主函数