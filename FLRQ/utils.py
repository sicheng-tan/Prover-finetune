import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def copy_small_files(src_folder, dst_folder, max_size_mb=10, info = False):
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name!= 'model.safetensors.index.json':
                src_file_path = os.path.join(root, file_name)
                # 获取文件大小，单位为字节
                file_size = os.path.getsize(src_file_path)
                # 将文件大小从字节转换为MB
                file_size_mb = file_size / (4* 1024 * 1024)

                # 如果文件小于指定的最大大小，则复制
                if file_size_mb < max_size_mb:
                    # 构建目标文件路径
                    relative_path = os.path.relpath(src_file_path, src_folder)
                    dst_file_path = os.path.join(dst_folder, relative_path)

                    # 确保目标文件夹存在
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                    # 复制文件
                    shutil.copy2(src_file_path, dst_file_path)  # 使用copy2以保留元数据
                    if info:
                        print(f"copy: {src_file_path} -> {dst_file_path}")
