#coding:utf-8
#!/usr/bin/env python
# 下载 ScanNet 公共数据集
# 使用 ./download-scannet.py 运行 (在 Windows 上使用 python download-scannet.py)
# -*- coding: utf-8 -*-
import argparse  # 导入 argparse 模块用于处理命令行参数
import os  # 导入 os 模块用于文件和目录操作
import urllib.request  # 导入 urllib.request 模块用于网络请求
import tempfile  # 导入 tempfile 模块用于创建临时文件

# 基础 URL，用于下载数据
BASE_URL = 'http://kaldir.vc.in.tum.de/scannet/'
# 服务条款 PDF 的 URL
TOS_URL = BASE_URL + 'ScanNet_TOS.pdf'
# 支持的文件类型列表
FILETYPES = [
    '.sens', 
    '.txt',
    '_vh_clean.ply', 
    '_vh_clean_2.ply',
    '_vh_clean.segs.json', 
    '_vh_clean_2.0.010000.segs.json',
    '.aggregation.json', 
    '_vh_clean.aggregation.json',
    '_vh_clean_2.labels.ply',
    '_2d-instance.zip', 
    '_2d-instance-filt.zip',
    '_2d-label.zip', 
    '_2d-label-filt.zip'
]
# 测试文件类型列表
FILETYPES_TEST = [
    '.sens', 
    '.txt', 
    '_vh_clean.ply', 
    '_vh_clean_2.ply'
]
# 预处理帧文件信息
PREPROCESSED_FRAMES_FILE = ['scannet_frames_25k.zip', '5.6GB']
# 测试帧文件信息
TEST_FRAMES_FILE = ['scannet_frames_test.zip', '610MB']
# 标签映射文件列表
LABEL_MAP_FILES = ['scannetv2-labels.combined.tsv', 'scannet-labels.combined.tsv']
# 发布版本列表
RELEASES = ['v2/scans', 'v1/scans']
# 任务版本列表
RELEASES_TASKS = ['v2/tasks', 'v1/tasks']
# 发布名称列表
RELEASES_NAMES = ['v2', 'v1']
# 当前发布版本
RELEASE = RELEASES[0]
# 当前任务版本
RELEASE_TASKS = RELEASES_TASKS[0]
# 当前发布名称
RELEASE_NAME = RELEASES_NAMES[0]
# 当前标签映射文件
LABEL_MAP_FILE = LABEL_MAP_FILES[0]
# 发布大小
RELEASE_SIZE = '1.2TB'
# v1 索引
V1_IDX = 1

# 获取发布的场景 ID
def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)  # 打开发布文件的 URL
    scans = []  # 初始化场景 ID 列表
    for scan_line in scan_lines:  # 遍历每一行
        scan_id = scan_line.decode('utf8').rstrip('\n')  # 解码并去除换行符
        scans.append(scan_id)  # 添加到场景 ID 列表
    return scans  # 返回场景 ID 列表

# 下载发布的场景
def download_release(release_scans, out_dir, file_types, use_v1_sens):
    if len(release_scans) == 0:  # 如果没有场景 ID
        return  # 直接返回
    print('Downloading ScanNet ' + RELEASE_NAME + ' release to ' + out_dir + '...')  # 打印下载信息
    for scan_id in release_scans:  # 遍历每个场景 ID
        scan_out_dir = os.path.join(out_dir, scan_id)  # 创建场景输出目录
        download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)  # 下载场景
    print('Downloaded ScanNet ' + RELEASE_NAME + ' release.')  # 打印下载完成信息

# 下载文件
def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)  # 获取输出文件的目录
    if not os.path.isdir(out_dir):  # 如果目录不存在
        os.makedirs(out_dir)  # 创建目录
    if not os.path.isfile(out_file):  # 如果文件不存在
        print('\t' + url + ' > ' + out_file)  # 打印下载信息
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)  # 创建临时文件
        f = os.fdopen(fh, 'w')  # 打开临时文件
        f.close()  # 关闭临时文件
        urllib.request.urlretrieve(url, out_file_tmp)  # 下载文件到临时文件
        os.rename(out_file_tmp, out_file)  # 重命名临时文件为目标文件
    else:
        print('WARNING: skipping download of existing file ' + out_file)  # 文件已存在，跳过下载

# 下载单个场景
def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    print('Downloading ScanNet ' + RELEASE_NAME + ' scan ' + scan_id + ' ...')  # 打印下载信息
    if not os.path.isdir(out_dir):  # 如果输出目录不存在
        os.makedirs(out_dir)  # 创建目录
    for ft in file_types:  # 遍历文件类型
        v1_sens = use_v1_sens and ft == '.sens'  # 判断是否使用 v1 的 .sens 文件
        # 构建下载 URL
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + scan_id + ft if not v1_sens else BASE_URL + RELEASES[
            V1_IDX] + '/' + scan_id + '/' + scan_id + ft
        out_file = out_dir + '/' + scan_id + ft  # 构建输出文件路径
        download_file(url, out_file)  # 下载文件
    print('Downloaded scan ' + scan_id)  # 打印下载完成信息

# 下载任务数据
def download_task_data(out_dir):
    print('Downloading ScanNet v1 task data...')  # 打印下载任务数据信息
    files = [
        LABEL_MAP_FILES[V1_IDX], 'obj_classification/data.zip',
        'obj_classification/trained_models.zip', 'voxel_labeling/data.zip',
        'voxel_labeling/trained_models.zip'
    ]
    for file in files:  # 遍历文件列表
        url = BASE_URL + RELEASES_TASKS[V1_IDX] + '/' + file  # 构建下载 URL
        localpath = os.path.join(out_dir, file)  # 构建本地文件路径
        localdir = os.path.dirname(localpath)  # 获取目录
        if not os.path.isdir(localdir):  # 如果目录不存在
            os.makedirs(localdir)  # 创建目录
        download_file(url, localpath)  # 下载文件
    print('Downloaded task data.')  # 打印下载完成信息

# 下载标签映射文件
def download_label_map(out_dir):
    print('Downloading ScanNet ' + RELEASE_NAME + ' label mapping file...')  # 打印下载标签映射信息
    files = [LABEL_MAP_FILE]  # 标签映射文件列表
    for file in files:  # 遍历文件列表
        url = BASE_URL + RELEASE_TASKS + '/' + file  # 构建下载 URL
        localpath = os.path.join(out_dir, file)  # 构建本地文件路径
        localdir = os.path.dirname(localpath)  # 获取目录
        if not os.path.isdir(localdir):  # 如果目录不存在
            os.makedirs(localdir)  # 创建目录
        download_file(url, localpath)  # 下载文件
    print('Downloaded ScanNet ' + RELEASE_NAME + ' label mapping file.')  # 打印下载完成信息

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Downloads ScanNet public data release.')  # 创建参数解析器
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')  # 输出目录参数
    parser.add_argument('--task_data', action='store_true', help='download task data (v1)')  # 下载任务数据参数
    parser.add_argument('--label_map', action='store_true', help='download label map file')  # 下载标签映射文件参数
    parser.add_argument('--v1', action='store_true', help='download ScanNet v1 instead of v2')  # 下载 v1 参数
    parser.add_argument('--id', help='specific scan id to download')  # 指定下载的场景 ID 参数
    parser.add_argument('--preprocessed_frames', action='store_true',
                        help='download preprocessed subset of ScanNet frames (' + PREPROCESSED_FRAMES_FILE[1] + ')')  # 下载预处理帧参数
    parser.add_argument('--test_frames_2d', action='store_true', help='download 2D test frames (' + TEST_FRAMES_FILE[
        1] + '; also included with whole dataset download)')  # 下载 2D 测试帧参数
    parser.add_argument('--type',
                        help='specific file type to download (.aggregation.json, .sens, .txt, _vh_clean.ply, _vh_clean_2.0.010000.segs.json, _vh_clean_2.ply, _vh_clean.segs.json, _vh_clean.aggregation.json, _vh_clean_2.labels.ply, _2d-instance.zip, _2d-instance-filt.zip, _2d-label.zip, _2d-label-filt.zip)')  # 指定下载的文件类型参数
    args = parser.parse_args()  # 解析参数

    print(
        'By pressing any key to continue you confirm that you have agreed to the ScanNet terms of use as described at:')  # 打印确认信息
    print(TOS_URL)  # 打印服务条款 URL
    print('***')  # 打印分隔符
    print('Press any key to continue, or CTRL-C to exit.')  # 提示用户继续或退出
    key = input('')  # 等待用户输入

    if args.v1:  # 如果选择下载 v1
        global RELEASE  # 声明全局变量
        global RELEASE_TASKS  # 声明全局变量
        global RELEASE_NAME  # 声明全局变量
        global LABEL_MAP_FILE  # 声明全局变量
        RELEASE = RELEASES[V1_IDX]  # 设置发布版本为 v1
        RELEASE_TASKS = RELEASES_TASKS[V1_IDX]  # 设置任务版本为 v1
        RELEASE_NAME = RELEASES_NAMES[V1_IDX]  # 设置发布名称为 v1
        LABEL_MAP_FILE = LABEL_MAP_FILES[V1_IDX]  # 设置标签映射文件为 v1

    release_file = BASE_URL + RELEASE + '.txt'  # 存放场景 ID 的文件
    release_scans = get_release_scans(release_file)  # 获取所有场景的 ID
    file_types = FILETYPES;  # 所有文件的后缀名
    release_test_file = BASE_URL + RELEASE + '_test.txt'  # 存放测试场景 ID 的文件
    release_test_scans = get_release_scans(release_test_file)  # 获取测试场景的 ID
    file_types_test = FILETYPES_TEST;  # 测试相关文件的后缀名
    out_dir_scans = os.path.join(args.out_dir, 'scans')  # 下载文件的子文件夹
    out_dir_test_scans = os.path.join(args.out_dir, 'scans_test')  # 下载文件的子文件夹
    out_dir_tasks = os.path.join(args.out_dir, 'tasks')  # 下载文件的子文件夹

    # 指定下载的文件类型
    if args.type:  # 如果指定了文件类型
        file_type = args.type  # 获取文件类型
        if file_type not in FILETYPES:  # 如果文件类型无效
            print('ERROR: Invalid file type: ' + file_type)  # 打印错误信息
            return  # 返回
        file_types = [file_type]  # 设置文件类型为指定类型
        if file_type in FILETYPES_TEST:  # 如果是测试文件类型
            file_types_test = [file_type]  # 设置测试文件类型
        else:
            file_types_test = []  # 清空测试文件类型
    if args.task_data:  # 如果选择下载任务数据
        download_task_data(out_dir_tasks)  # 下载任务数据
    elif args.label_map:  # 如果选择下载标签映射文件
        download_label_map(args.out_dir)  # 下载标签映射文件
    elif args.preprocessed_frames:  # 如果选择下载预处理帧
        if args.v1:  # 如果选择 v1
            print('ERROR: Preprocessed frames only available for ScanNet v2')  # 打印错误信息
        print('You are downloading the preprocessed subset of frames ' + PREPROCESSED_FRAMES_FILE[
            0] + ' which requires ' + PREPROCESSED_FRAMES_FILE[1] + ' of space.')  # 打印下载信息
        download_file(os.path.join(BASE_URL, RELEASE_TASKS, PREPROCESSED_FRAMES_FILE[0]),
                      os.path.join(out_dir_tasks, PREPROCESSED_FRAMES_FILE[0]))  # 下载预处理帧
    elif args.test_frames_2d:  # 如果选择下载 2D 测试帧
        if args.v1:  # 如果选择 v1
            print('ERROR: 2D test frames only available for ScanNet v2')  # 打印错误信息
        print('You are downloading the 2D test set ' + TEST_FRAMES_FILE[0] + ' which requires ' + TEST_FRAMES_FILE[
            1] + ' of space.')  # 打印下载信息
        download_file(os.path.join(BASE_URL, RELEASE_TASKS, TEST_FRAMES_FILE[0]),
                      os.path.join(out_dir_tasks, TEST_FRAMES_FILE[0]))  # 下载 2D 测试帧
    elif args.id:  # 如果选择下载单个场景
        scan_id = args.id  # 获取场景 ID
        is_test_scan = scan_id in release_test_scans  # 判断是否为测试场景
        if scan_id not in release_scans and (not is_test_scan or args.v1):  # 如果场景 ID 无效
            print('ERROR: Invalid scan id: ' + scan_id)  # 打印错误信息
        else:
            out_dir = os.path.join(out_dir_scans, scan_id) if not is_test_scan else os.path.join(out_dir_test_scans,
                                                                                                 scan_id)  # 设置输出目录
            scan_file_types = file_types if not is_test_scan else file_types_test  # 设置文件类型
            use_v1_sens = not is_test_scan  # 判断是否使用 v1 的 .sens 文件
            if not is_test_scan and not args.v1 and '.sens' in scan_file_types:  # 如果不是测试场景且不是 v1
                print(
                    'Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press \'n\' to exclude downloading .sens files for each scan')  # 提示用户
                key = input('')  # 等待用户输入
                if key.strip().lower() == 'n':  # 如果用户选择不下载 .sens 文件
                    scan_file_types.remove('.sens')  # 从文件类型中移除 .sens
            download_scan(scan_id, out_dir, scan_file_types, use_v1_sens)  # 下载场景
    else:  # 下载整个发布
        if len(file_types) == len(FILETYPES):  # 如果下载所有文件类型
            print(
                'WARNING: You are downloading the entire ScanNet ' + RELEASE_NAME + ' release which requires ' + RELEASE_SIZE + ' of space.')  # 打印警告信息
        else:
            print('WARNING: You are downloading all ScanNet ' + RELEASE_NAME + ' scans of type ' + file_types[0])  # 打印警告信息
        print(
            'Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')  # 提示用户
        print('***')  # 打印分隔符
        print('Press any key to continue, or CTRL-C to exit.')  # 提示用户继续或退出
        key = input('')  # 等待用户输入
        if not args.v1 and '.sens' in file_types:  # 如果不是 v1 且文件类型中包含 .sens
            print(
                'Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press \'n\' to exclude downloading .sens files for each scan')  # 提示用户
            key = input('')  # 等待用户输入
            if key.strip().lower() == 'n':  # 如果用户选择不下载 .sens 文件
                file_types.remove('.sens')  # 从文件类型中移除 .sens
        download_release(release_scans, out_dir_scans, file_types, use_v1_sens=True)  # 下载发布的场景
        if not args.v1:  # 如果不是 v1
            download_label_map(args.out_dir)  # 下载标签映射文件
            download_release(release_test_scans, out_dir_test_scans, file_types_test, use_v1_sens=False)  # 下载测试场景
            download_file(os.path.join(BASE_URL, RELEASE_TASKS, TEST_FRAMES_FILE[0]),
                          os.path.join(out_dir_tasks, TEST_FRAMES_FILE[0]))  # 下载测试帧文件

# 程序入口
if __name__ == "__main__": main()  # 调用主函数