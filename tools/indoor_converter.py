# Modified from mmdetection3d/tools/dataset_converters/indoor_converter.py
# We just support ScanNet 200.
import os  # 导入os模块，用于处理文件和目录路径

import mmengine  # 导入mmengine模块，用于模型训练和数据处理

from scannet_data_utils import ScanNetData  # 从scannet_data_utils导入ScanNetData类，用于处理ScanNet数据


def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',  # pkl文件的前缀，默认为'sunrgbd'
                            save_path=None,  # pkl文件保存路径，默认为None
                            use_v1=False,  # 是否使用版本1，默认为False
                            workers=4):  # 使用的线程数，默认为4
    """Create indoor information file.  # 创建室内信息文件

    Get information of the raw data and save it to the pkl file.  # 获取原始数据的信息并保存到pkl文件

    Args:  # 参数说明
        data_path (str): Path of the data.  # 数据路径
        pkl_prefix (str, optional): Prefix of the pkl to be saved.  # 要保存的pkl文件前缀
            Default: 'sunrgbd'.  # 默认值为'sunrgbd'
        save_path (str, optional): Path of the pkl to be saved. Default: None.  # pkl文件保存路径，默认值为None
        use_v1 (bool, optional): Whether to use v1. Default: False.  # 是否使用版本1，默认值为False
        workers (int, optional): Number of threads to be used. Default: 4.  # 使用的线程数，默认值为4
    """
    assert os.path.exists(data_path)  # 确保数据路径存在
    # 确保前缀在支持的选项中
    assert pkl_prefix in ['scannet', 'scannet200'], f'unsupported indoor dataset {pkl_prefix}'  # 抛出不支持的数据集异常
    save_path = data_path if save_path is None else save_path  # 如果没有提供保存路径，则使用数据路径
    assert os.path.exists(save_path)  # 确保保存路径存在

    # generate infos for both detection and segmentation task  # 生成检测和分割任务的信息
    train_filename = os.path.join(  # 训练文件名
        save_path, f'{pkl_prefix}_oneformer3d_infos_train.pkl')  # 拼接保存路径和文件名
    val_filename = os.path.join(  # 验证文件名
        save_path, f'{pkl_prefix}_oneformer3d_infos_val.pkl')  # 拼接保存路径和文件名
    test_filename = os.path.join(  # 测试文件名
        save_path, f'{pkl_prefix}_oneformer3d_infos_test.pkl')  # 拼接保存路径和文件名
    if pkl_prefix == 'scannet':  # 如果前缀是'scannet'
        # ScanNet has a train-val-test split  # ScanNet有训练-验证-测试的划分
        train_dataset = ScanNetData(root_path=data_path, split='train')  # 创建训练数据集
        val_dataset = ScanNetData(root_path=data_path, split='val')  # 创建验证数据集
        test_dataset = ScanNetData(root_path=data_path, split='test')  # 创建测试数据集
    else:  # 否则为ScanNet200
        # ScanNet has a train-val-test split  # ScanNet200也有训练-验证-测试的划分
        train_dataset = ScanNetData(root_path=data_path, split='train',  # 创建训练数据集
                                    scannet200=True, save_path=save_path)  # 指定为ScanNet200
        val_dataset = ScanNetData(root_path=data_path, split='val',  # 创建验证数据集
                                    scannet200=True, save_path=save_path)  # 指定为ScanNet200
        test_dataset = ScanNetData(root_path=data_path, split='test',  # 创建测试数据集
                                    scannet200=True, save_path=save_path)  # 指定为ScanNet200

    infos_train = train_dataset.get_infos(  # 获取训练数据集的信息
        num_workers=workers, has_label=True)  # 指定线程数并要求有标签
    mmengine.dump(infos_train, train_filename, 'pkl')  # 将训练信息保存为pkl文件
    print(f'{pkl_prefix} info train file is saved to {train_filename}')  # 打印保存路径

    infos_val = val_dataset.get_infos(  # 获取验证数据集的信息
        num_workers=workers, has_label=True)  # 指定线程数并要求有标签
    mmengine.dump(infos_val, val_filename, 'pkl')  # 将验证信息保存为pkl文件
    print(f'{pkl_prefix} info val file is saved to {val_filename}')  # 打印保存路径

    infos_test = test_dataset.get_infos(  # 获取测试数据集的信息
        num_workers=workers, has_label=False)  # 指定线程数且不要求有标签
    mmengine.dump(infos_test, test_filename, 'pkl')  # 将测试信息保存为pkl文件
    print(f'{pkl_prefix} info test file is saved to {test_filename}')  # 打印保存路径