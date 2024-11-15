# Copyright (c) OpenMMLab. All rights reserved.
import os  # 导入操作系统相关的模块
from concurrent import futures as futures  # 导入并发执行的模块
from os import path as osp  # 导入路径操作模块并重命名为osp

import mmengine  # 导入mmengine库
import numpy as np  # 导入numpy库并重命名为np


class ScanNetData(object):  # 定义ScanNetData类
    """ScanNet数据类。
    生成用于scannet_converter的scannet信息。

    参数:
        root_path (str): 原始数据的根路径。
        split (str, optional): 数据的分割类型。默认值: 'train'。
        scannet200 (bool): 如果为True则为ScanNet200，否则为ScanNet。
        save_path (str, optional): 输出目录。
    """

    def __init__(self, root_path, split='train', scannet200=False, save_path=None):  # 初始化方法
        self.root_dir = root_path  # 设置根目录
        self.save_path = root_path if save_path is None else save_path  # 设置保存路径
        self.split = split  # 设置数据分割类型
        self.split_dir = osp.join(root_path)  # 连接根路径
        self.scannet200 = scannet200  # 设置是否为ScanNet200
        if self.scannet200:  # 如果是ScanNet200
            self.classes = [  # 定义类别列表
                'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
                'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
                'toilet', 'bookshelf', 'monitor', 'curtain', 'book',
                'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
                'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand',
                'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling',
                'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
                'backpack', 'toilet paper', 'printer', 'tv stand',
                'whiteboard', 'blanket', 'shower curtain', 'trash can',
                'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress'
            ]
            self.cat_ids = np.array([  # 定义类别ID数组
                2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
                22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40,
                41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
                59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116,
                118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138,
                139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163,
                165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
                202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261,
                264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356,
                370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581,
                609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
                1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180,
                1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190,
                1191
            ])
        else:  # 如果不是ScanNet200
            self.classes = [  # 定义类别列表
                'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin'
            ]
            self.cat_ids = np.array([  # 定义类别ID数组
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
            ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}  # 创建类别到标签的映射
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}  # 创建标签到类别的映射
        self.cat_ids2class = {  # 创建类别ID到类别的映射
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']  # 确保分割类型有效
        split_file = osp.join(self.root_dir, 'meta_data', f'scannetv2_{split}.txt')  # 构建分割文件路径
        mmengine.check_file_exist(split_file)  # 检查分割文件是否存在
        self.sample_id_list = mmengine.list_from_file(split_file)  # 从文件中读取样本ID列表
        self.test_mode = (split == 'test')  # 设置测试模式

    def __len__(self):  # 定义获取样本数量的方法
        return len(self.sample_id_list)  # 返回样本ID列表的长度

    def get_aligned_box_label(self, idx):  # 获取对齐的边界框标签
        box_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_aligned_bbox.npy')  # 构建边界框文件路径
        mmengine.check_file_exist(box_file)  # 检查边界框文件是否存在
        return np.load(box_file)  # 加载并返回边界框数据

    def get_unaligned_box_label(self, idx):  # 获取未对齐的边界框标签
        box_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_unaligned_bbox.npy')  # 构建未对齐边界框文件路径
        mmengine.check_file_exist(box_file)  # 检查未对齐边界框文件是否存在
        return np.load(box_file)  # 加载并返回未对齐边界框数据

    def get_axis_align_matrix(self, idx):  # 获取轴对齐矩阵
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_axis_align_matrix.npy')  # 构建矩阵文件路径
        mmengine.check_file_exist(matrix_file)  # 检查矩阵文件是否存在
        return np.load(matrix_file)  # 加载并返回轴对齐矩阵数据

    def get_images(self, idx):  # 获取图像路径
        paths = []  # 初始化图像路径列表
        path = osp.join(self.root_dir, 'posed_images', idx)  # 构建图像目录路径
        for file in sorted(os.listdir(path)):  # 遍历目录中的文件
            if file.endswith('.jpg'):  # 如果文件是jpg格式
                paths.append(osp.join('posed_images', idx, file))  # 添加图像路径到列表
        return paths  # 返回图像路径列表

    def get_extrinsics(self, idx):  # 获取外部参数
        extrinsics = []  # 初始化外部参数列表
        path = osp.join(self.root_dir, 'posed_images', idx)  # 构建图像目录路径
        for file in sorted(os.listdir(path)):  # 遍历目录中的文件
            if file.endswith('.txt') and not file == 'intrinsic.txt':  # 如果文件是txt格式且不是内参文件
                extrinsics.append(np.loadtxt(osp.join(path, file)))  # 加载外部参数并添加到列表
        return extrinsics  # 返回外部参数列表

    def get_intrinsics(self, idx):  # 获取内部参数
        matrix_file = osp.join(self.root_dir, 'posed_images', idx, 'intrinsic.txt')  # 构建内部参数文件路径
        mmengine.check_file_exist(matrix_file)  # 检查内部参数文件是否存在
        return np.loadtxt(matrix_file)  # 加载并返回内部参数数据

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):  # 获取数据的信息
        """获取数据的信息。

        此方法从原始数据中获取信息。

        参数:
            num_workers (int, optional): 使用的线程数量。默认值: 4。
            has_label (bool, optional): 数据是否有标签。默认值: True。
            sample_id_list (list[int], optional): 样本的索引列表。默认值: None。

        返回:
            infos (list[dict]): 原始数据的信息。
        """

        def process_single_scene(sample_idx):  # 处理单个场景
            print(f'{self.split} sample_idx: {sample_idx}')  # 打印当前样本索引
            info = dict()  # 初始化信息字典
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}  # 设置点云信息
            info['point_cloud'] = pc_info  # 将点云信息添加到信息字典
            pts_filename = osp.join(self.root_dir, 'scannet_instance_data', f'{sample_idx}_vert.npy')  # 构建点云文件路径
            
            # TODO: 等不起了，增加如果点云数据文件不存在，就跳过。
            if not os.path.exists(pts_filename):  # 检查点云文件是否存在
                print(f'Exist. File is not exist: {pts_filename}')  # 打印文件不存在的消息
                return;  # 跳过当前处理
            
            points = np.load(pts_filename)  # 加载点云数据
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))  # 创建保存点云的目录
            points.tofile(  # 将点云数据保存为二进制文件
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')  # 添加点云路径到信息字典

            sp_filename = osp.join(self.root_dir, 'scannet_instance_data', f'{sample_idx}_sp_label.npy')  # 构建超点标签文件路径
            super_points = np.load(sp_filename)  # 加载超点数据
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))  # 创建保存超点的目录
            super_points.tofile(  # 将超点数据保存为二进制文件
                osp.join(self.save_path, 'super_points', f'{sample_idx}.bin'))
            info['super_pts_path'] = osp.join('super_points', f'{sample_idx}.bin')  # 添加超点路径到信息字典

            # 更新RGB图像路径（如果存在）
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):  # 检查图像目录是否存在
                info['intrinsics'] = self.get_intrinsics(sample_idx)  # 获取内部参数并添加到信息字典
                all_extrinsics = self.get_extrinsics(sample_idx)  # 获取所有外部参数
                all_img_paths = self.get_images(sample_idx)  # 获取所有图像路径
                # ScanNet中的某些姿态是无效的
                extrinsics, img_paths = [], []  # 初始化有效外部参数和图像路径列表
                for extrinsic, img_path in zip(all_extrinsics, all_img_paths):  # 遍历外部参数和图像路径
                    if np.all(np.isfinite(extrinsic)):  # 检查外部参数是否有效
                        img_paths.append(img_path)  # 添加有效图像路径
                        extrinsics.append(extrinsic)  # 添加有效外部参数
                info['extrinsics'] = extrinsics  # 添加外部参数到信息字典
                info['img_paths'] = img_paths  # 添加图像路径到信息字典

            if not self.test_mode:  # 如果不是测试模式
                pts_instance_mask_path = osp.join(  # 构建点实例掩码文件路径
                    self.root_dir, 'scannet_instance_data', f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(  # 构建点语义掩码文件路径
                    self.root_dir, 'scannet_instance_data', f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(  # 加载点实例掩码并转换为int64类型
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(  # 加载点语义掩码并转换为int64类型
                    np.int64)

                mmengine.mkdir_or_exist(  # 创建保存实例掩码的目录
                    osp.join(self.save_path, 'instance_mask'))
                mmengine.mkdir_or_exist(  # 创建保存语义掩码的目录
                    osp.join(self.save_path, 'semantic_mask'))

                pts_instance_mask.tofile(  # 将点实例掩码保存为二进制文件
                    osp.join(self.save_path, 'instance_mask', f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(  # 将点语义掩码保存为二进制文件
                    osp.join(self.save_path, 'semantic_mask', f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(  # 添加点实例掩码路径到信息字典
                    'instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(  # 添加点语义掩码路径到信息字典
                    'semantic_mask', f'{sample_idx}.bin')

            if has_label:  # 如果有标签
                annotations = {}  # 初始化注释字典
                # box的形状为[k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)  # 获取对齐的边界框标签
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)  # 获取未对齐的边界框标签
                annotations['gt_num'] = aligned_box_label.shape[0]  # 获取对齐边界框的数量
                if annotations['gt_num'] != 0:  # 如果有对齐边界框
                    aligned_box = aligned_box_label[:, :-1]  # 获取对齐边界框的位置信息
                    unaligned_box = unaligned_box_label[:, :-1]  # 获取未对齐边界框的位置信息
                    classes = aligned_box_label[:, -1]  # 获取对齐边界框的类别信息
                    annotations['name'] = np.array([  # 获取类别名称
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # 默认名称用于对齐边界框以兼容性
                    # 我们还保存未对齐边界框信息和标记名称
                    annotations['location'] = aligned_box[:, :3]  # 获取对齐边界框的位置
                    annotations['dimensions'] = aligned_box[:, 3:6]  # 获取对齐边界框的尺寸
                    annotations['gt_boxes_upright_depth'] = aligned_box  # 保存对齐边界框
                    annotations['unaligned_location'] = unaligned_box[:, :3]  # 获取未对齐边界框的位置
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]  # 获取未对齐边界框的尺寸
                    annotations['unaligned_gt_boxes_upright_depth'] = unaligned_box  # 保存未对齐边界框
                    annotations['index'] = np.arange(  # 创建边界框索引
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([  # 获取未对齐边界框的类别
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)  # 获取轴对齐矩阵
                annotations['axis_align_matrix'] = axis_align_matrix  # 添加轴对齐矩阵到注释字典
                info['annos'] = annotations  # 将注释添加到信息字典
            return info  # 返回信息字典

        # 设置样本ID列表
        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list  # 如果没有提供，则使用默认样本ID列表
        with futures.ThreadPoolExecutor(num_workers) as executor:  # 创建线程池执行器
            infos = executor.map(process_single_scene, sample_id_list)  # 并发处理每个样本
        return list(infos)  # 返回处理结果列表