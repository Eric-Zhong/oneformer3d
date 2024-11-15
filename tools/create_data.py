# Copyright (c) OpenMMLab. All rights reserved.
import argparse  # 导入argparse模块，用于处理命令行参数
from os import path as osp  # 从os模块导入path并重命名为osp，用于路径操作

from indoor_converter import create_indoor_info_file  # 从indoor_converter模块导入create_indoor_info_file函数
from update_infos_to_v2 import update_pkl_infos  # 从update_infos_to_v2模块导入update_pkl_infos函数


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """准备scannet数据集的信息文件。

    Args:
        root_path (str): 数据集根路径。
        info_prefix (str): 信息文件名的前缀。
        out_dir (str): 生成的信息文件的输出目录。
        workers (int): 使用的线程数量。
    """
    create_indoor_info_file(root_path, info_prefix, out_dir, workers=workers)  # 创建室内信息文件
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')  # 训练集信息文件路径
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')  # 验证集信息文件路径
    info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')  # 测试集信息文件路径
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)  # 更新训练集信息
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)  # 更新验证集信息
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)  # 更新测试集信息


parser = argparse.ArgumentParser(description='数据转换器参数解析器')  # 创建参数解析器
parser.add_argument('dataset', metavar='kitti', help='数据集名称')  # 添加数据集名称参数
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='指定数据集的根路径')  # 添加根路径参数
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='信息pkl的名称')  # 添加输出目录参数
parser.add_argument('--extra-tag', type=str, default='kitti')  # 添加额外标签参数
parser.add_argument(
    '--workers', type=int, default=4, help='使用的线程数量')  # 添加工作线程数量参数
args = parser.parse_args()  # 解析命令行参数

if __name__ == '__main__':  # 如果该文件是主程序
    from mmdet3d.utils import register_all_modules  # 从mmdet3d.utils模块导入register_all_modules函数
    register_all_modules()  # 注册所有模块

    if args.dataset in ('scannet', 'scannet200'):  # 如果数据集是scannet或scannet200
        scannet_data_prep(  # 调用scannet_data_prep函数
            root_path=args.root_path,  # 传入根路径
            info_prefix=args.extra_tag,  # 传入信息前缀
            out_dir=args.out_dir,  # 传入输出目录
            workers=args.workers)  # 传入工作线程数量
    else:  # 如果数据集不在支持的范围内
        raise NotImplementedError(f'不支持{args.dataset}数据集。')  # 抛出未实现错误