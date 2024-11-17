_base_ = [
    'mmdet3d::_base_/default_runtime.py',  # 基础运行时配置
    'mmdet3d::_base_/datasets/scannet-seg.py'  # 基础数据集配置
]
custom_imports = dict(imports=['oneformer3d'])  # 自定义导入的模块

# model settings
num_channels = 32  # 模型中通道的数量
num_instance_classes = 18  # 实例类别的数量
num_semantic_classes = 20  # 语义类别的数量

model = dict(
    type='ScanNetOneFormer3D',  # 模型类型
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),  # 数据预处理器类型
    in_channels=6,  # 输入通道数
    num_channels=num_channels,  # 模型中通道的数量
    voxel_size=0.02,  # 体素大小
    num_classes=num_instance_classes,  # 实例类别数量
    min_spatial_shape=128,  # 最小空间形状
    query_thr=0.5,  # 查询阈值
    backbone=dict(
        type='SpConvUNet',  # 主干网络类型
        num_planes=[num_channels * (i + 1) for i in range(5)],  # 每层的通道数
        return_blocks=True),  # 是否返回块
    decoder=dict(
        type='ScanNetQueryDecoder',  # 解码器类型
        num_layers=6,  # 解码器层数
        num_instance_queries=0,  # 实例查询数量
        num_semantic_queries=0,  # 语义查询数量
        num_instance_classes=num_instance_classes,  # 实例类别数量
        num_semantic_classes=num_semantic_classes,  # 语义类别数量
        num_semantic_linears=1,  # 语义线性层数量
        in_channels=32,  # 输入通道数
        d_model=256,  # 模型维度
        num_heads=8,  # 注意力头数量
        hidden_dim=1024,  # 隐藏层维度
        dropout=0.0,  # dropout比率
        activation_fn='gelu',  # 激活函数
        iter_pred=True,  # 是否进行迭代预测
        attn_mask=True,  # 是否使用注意力掩码
        fix_attention=True,  # 是否固定注意力
        objectness_flag=False),  # 是否使用物体性标志
    criterion=dict(
        type='ScanNetUnifiedCriterion',  # 统一损失函数类型
        num_semantic_classes=num_semantic_classes,  # 语义类别数量
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',  # 语义损失函数类型
            ignore_index=num_semantic_classes,  # 忽略的索引
            loss_weight=0.2),  # 损失权重
        inst_criterion=dict(
            type='InstanceCriterion',  # 实例损失函数类型
            matcher=dict(
                type='SparseMatcher',  # 匹配器类型
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),  # 查询分类损失
                    dict(type='MaskBCECost', weight=1.0),  # 掩码二元交叉熵损失
                    dict(type='MaskDiceCost', weight=1.0)],  # 掩码Dice损失
                topk=1),  # 选择的前k个
            loss_weight=[0.5, 1.0, 1.0, 0.5],  # 损失权重
            num_classes=num_instance_classes,  # 实例类别数量
            non_object_weight=0.1,  # 非物体权重
            fix_dice_loss_weight=True,  # 是否固定Dice损失权重
            iter_matcher=True,  # 是否使用迭代匹配器
            fix_mean_loss=True)),  # 是否固定均值损失
    train_cfg=dict(),  # 训练配置
    test_cfg=dict(
        topk_insts=600,  # 测试时的前k个实例
        inst_score_thr=0.0,  # 实例分数阈值
        pan_score_thr=0.5,  # 全景分数阈值
        npoint_thr=100,  # 点数阈值
        obj_normalization=True,  # 是否进行物体归一化
        sp_score_thr=0.4,  # 超点分数阈值
        nms=True,  # 是否使用非极大值抑制
        matrix_nms_kernel='linear',  # 矩阵NMS核类型
        stuff_classes=[0, 1]))  # 杂物类别

# dataset settings
dataset_type = 'ScanNetSegDataset_'  # 数据集类型
data_prefix = dict(
    pts='points',  # 点云数据
    pts_instance_mask='instance_mask',  # 实例掩码
    pts_semantic_mask='semantic_mask',  # 语义掩码
    sp_pts_mask='super_points')  # 超点掩码

train_pipeline = [
    dict(
        type='LoadPointsFromFile',  # 从文件加载点云
        coord_type='DEPTH',  # 坐标类型
        shift_height=False,  # 是否偏移高度
        use_color=True,  # 是否使用颜色信息
        load_dim=6,  # 加载维度
        use_dim=[0, 1, 2, 3, 4, 5]),  # 使用的维度
    dict(
        type='LoadAnnotations3D_',  # 加载3D注释
        with_bbox_3d=False,  # 是否包含3D边界框
        with_label_3d=False,  # 是否包含3D标签
        with_mask_3d=True,  # 是否包含3D掩码
        with_seg_3d=True,  # 是否包含3D分割
        with_sp_mask_3d=True),  # 是否包含3D超点掩码
    dict(type='PointSegClassMapping'),  # 点分割类别映射
    dict(
        type='RandomFlip3D',  # 随机3D翻转
        sync_2d=False,  # 是否同步2D翻转
        flip_ratio_bev_horizontal=0.5,  # BEV水平翻转比率
        flip_ratio_bev_vertical=0.5),  # BEV垂直翻转比率
    dict(
        type='GlobalRotScaleTrans',  # 全局旋转缩放变换
        rot_range=[-3.14, 3.14],  # 旋转范围
        scale_ratio_range=[0.8, 1.2],  # 缩放比率范围
        translation_std=[0.1, 0.1, 0.1],  # 平移标准差
        shift_height=False),  # 是否偏移高度
    dict(
        type='NormalizePointsColor_',  # 归一化点云颜色
        color_mean=[127.5, 127.5, 127.5]),  # 颜色均值
    dict(
        type='AddSuperPointAnnotations',  # 添加超点注释
        num_classes=num_semantic_classes,  # 语义类别数量
        stuff_classes=[0, 1],  # 杂物类别
        merge_non_stuff_cls=False),  # 是否合并非杂物类别
    dict(
        type='ElasticTransfrom',  # 弹性变换
        gran=[6, 20],  # 粒度
        mag=[40, 160],  # 变换幅度
        voxel_size=0.02,  # 体素大小
        p=0.5),  # 变换概率
    dict(
        type='Pack3DDetInputs_',  # 打包3D检测输入
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])  # 输入键
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',  # 从文件加载点云
        coord_type='DEPTH',  # 坐标类型
        shift_height=False,  # 是否偏移高度
        use_color=True,  # 是否使用颜色信息
        load_dim=6,  # 加载维度
        use_dim=[0, 1, 2, 3, 4, 5]),  # 使用的维度
    dict(
        type='LoadAnnotations3D_',  # 加载3D注释
        with_bbox_3d=False,  # 是否包含3D边界框
        with_label_3d=False,  # 是否包含3D标签
        with_mask_3d=True,  # 是否包含3D掩码
        with_seg_3d=True,  # 是否包含3D分割
        with_sp_mask_3d=True),  # 是否包含3D超点掩码
    dict(type='PointSegClassMapping'),  # 点分割类别映射
    dict(
        type='MultiScaleFlipAug3D',  # 多尺度翻转增强
        img_scale=(1333, 800),  # 图像尺度
        pts_scale_ratio=1,  # 点云尺度比率
        flip=False,  # 是否翻转
        transforms=[
            dict(
                type='NormalizePointsColor_',  # 归一化点云颜色
                color_mean=[127.5, 127.5, 127.5]),  # 颜色均值
            dict(
                type='AddSuperPointAnnotations',  # 添加超点注释
                num_classes=num_semantic_classes,  # 语义类别数量
                stuff_classes=[0, 1],  # 杂物类别
                merge_non_stuff_cls=False),  # 是否合并非杂物类别
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])  # 打包3D检测输入
]

# run settings
train_dataloader = dict(
    batch_size=4,  # 批次大小
    num_workers=6,  # 工作线程数量
    dataset=dict(
        type=dataset_type,  # 数据集类型
        ann_file='scannet_oneformer3d_infos_train.pkl',  # 训练注释文件
        data_prefix=data_prefix,  # 数据前缀
        pipeline=train_pipeline,  # 训练管道
        ignore_index=num_semantic_classes,  # 忽略的索引
        scene_idxs=None,  # 场景索引
        test_mode=False))  # 是否为测试模式
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,  # 数据集类型
        ann_file='scannet_oneformer3d_infos_val.pkl',  # 验证注释文件
        data_prefix=data_prefix,  # 数据前缀
        pipeline=test_pipeline,  # 测试管道
        ignore_index=num_semantic_classes,  # 忽略的索引
        test_mode=True))  # 是否为测试模式
test_dataloader = val_dataloader  # 测试数据加载器与验证数据加载器相同

class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
    'bathtub', 'otherfurniture']  # 类别名称
class_names += ['unlabeled']  # 添加未标记类别
label2cat = {i: name for i, name in enumerate(class_names)}  # 类别标签映射
metric_meta = dict(
    label2cat=label2cat,  # 类别标签映射
    ignore_index=[num_semantic_classes],  # 忽略的索引
    classes=class_names,  # 类别名称
    dataset_name='ScanNet')  # 数据集名称

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]  # 语义映射
inst_mapping = sem_mapping[2:]  # 实例映射
val_evaluator = dict(
    type='UnifiedSegMetric',  # 统一分割评估器类型
    stuff_class_inds=[0, 1],  # 杂物类别索引
    thing_class_inds=list(range(2, num_semantic_classes)),  # 物体类别索引
    min_num_points=1,  # 最小点数
    id_offset=2**16,  # ID偏移量
    sem_mapping=sem_mapping,  # 语义映射
    inst_mapping=inst_mapping,  # 实例映射
    metric_meta=metric_meta)  # 评估元数据
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同

optim_wrapper = dict(
    type='OptimWrapper',  # 优化器包装器类型
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),  # 优化器类型及参数
    clip_grad=dict(max_norm=10, norm_type=2))  # 梯度裁剪设置

param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9)  # 学习率调度器设置

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]  # 自定义钩子设置
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16))  # 默认钩子设置

load_from = 'work_dirs/tmp/sstnet_scannet.pth'  # 加载预训练模型路径

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 基于周期的训练循环类型
    max_epochs=512,  # 最大训练周期
    dynamic_intervals=[(1, 16), (512 - 16, 1)])  # 动态间隔设置
val_cfg = dict(type='ValLoop')  # 验证循环配置
test_cfg = dict(type='TestLoop')  # 测试循环配置