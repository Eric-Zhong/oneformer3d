'''
fix_spconv_checkpoint.py是 mmdetection3d 中的一个脚本文件，主要用于修复旧版本 spconv 库生成的模型检查点（checkpoint），
以便能够在新版本的 spconv 库下正确加载模型。
'''
import argparse  # 导入argparse模块，用于处理命令行参数
import torch  # 导入torch库，用于深度学习和张量操作


if __name__ == '__main__':  # 检查是否为主程序入口
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--in-path', type=str, required=True)  # 添加输入路径参数，必需
    parser.add_argument('--out-path', type=str, required=True)  # 添加输出路径参数，必需
    args = parser.parse_args()  # 解析命令行参数并存储在args中

    checkpoint = torch.load(args.in_path)  # 从指定的输入路径加载检查点文件
    key = 'state_dict'  # 定义键名为'state_dict'，用于访问模型状态字典
    for layer in checkpoint[key]:  # 遍历状态字典中的每一层
        # 检查层名是否以'unet'或'input_conv'开头
        # 检查层名是否以'weight'结尾
        if (layer.startswith('unet') or layer.startswith('input_conv')) \
            and layer.endswith('weight') \
                and len(checkpoint[key][layer].shape) == 5:  # 检查权重的形状是否为5维
            checkpoint[key][layer] = checkpoint[key][layer].permute(1, 2, 3, 4, 0)  # 重新排列权重的维度
    torch.save(checkpoint, args.out_path)  # 将修改后的检查点保存到指定的输出路径