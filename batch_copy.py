# 批量复制图片
import os

import shutil

def copy_remote(images_path):
    new_file_path = os.path.join(images_path, "remotes")
    if os.path.exists(new_file_path) is False:
        os.makedirs(new_file_path)

    for file in os.listdir(images_path):
        if file.endswith("tif"):
            file_path = os.path.join(images_path, file)
            new_file_name = os.path.join(new_file_path, os.path.basename(file).split('.')[0] + '.png')
            shutil.copyfile(file_path,new_file_name)

def copy_dem(images_path):
    new_file_path = os.path.join(images_path, "dems")
    if os.path.exists(new_file_path) is False:
        os.makedirs(new_file_path)

    for file in os.listdir(images_path):
        if file.endswith("tif"):
            file_path = os.path.join(images_path, file)
            new_file_name = os.path.join(new_file_path, os.path.basename(file).split('.')[0] + '.tif')
            shutil.copyfile(file_path,new_file_name)


if __name__ == "__main__":
    copy_remote(r"D:\village_data\侗族传统村落图数据\村落影像图")
    # copy_dem(r"D:\village_data\侗族传统村落图数据\村落dem")