"""
    批量运行村落边界提取和几何中心计算
"""
import os

import numpy as np
from PIL import Image
import cv2 as cv
from config.params import *

def batch_process_village_edge(batch_images_path):
    """
    批量提取村落边界
    :param batch_images_path:
    :return:
    """
    from Up_to_Domain.village_edge_dev.interfaces import edge_detect

    center_point_list = []
    # 获取文件列表
    for file in os.listdir(batch_images_path):
        file_path = os.path.join(batch_images_path,file)

        [x,y] = edge_detect.interface_villageEdgeDect(file_path)
        center_point_list.append([x,y])

    np.savetxt(os.path.join(batch_images_path,"centerpoints.txt"),center_point_list,fmt="%d %d")

def batch_processing_village_accessableArea(batch_images_path):
    '''
    批量计算村落可达域
    :param batch_images_path:
    :return:
    '''

    from Up_to_Domain.interfaces.up_to_domain import interface_uptodomain

    remote_path = os.path.join(batch_images_path,"remotes")
    dem_path = os.path.join(batch_images_path, "dems")

    # 获取村落名称列表
    village_name_lists = list(map(lambda x: x.split('remote')[0],os.listdir(remote_path)))

    # 村落dem路径列表和remote路径列表
    village_dem_lists = list(map(lambda x: os.path.join(dem_path,x + 'dem.tif'),village_name_lists))
    village_remote_lists = list(map(lambda x: os.path.join(remote_path, x + 'remote.png'), village_name_lists))

    length = len(village_name_lists)
    for i in range(length):
        interface_uptodomain(village_dem_lists[i],village_remote_lists[i],timelimit=60,demAxis=10)


def batch_processing_ascensionpoints(batch_images_path):
    '''
    批量计算登临点
    :param batch_images_path:
    :return:
    '''

    from ascensionpoint_generate.interfaces.interface_visionpoint import interface_cal_ascensionpoint

    dem_path = os.path.join(batch_images_path, "dems")

    for dem in os.listdir(dem_path):
        dem_file = os.path.join(dem_path,dem)
        interface_cal_ascensionpoint(dem_file,denglin_num=50,time_limit=60,divideSpace=40,demAxis=10)

def batch_processing_drawpics(ascension_results_path, dems_path, remotes_path):
    '''
    传入参数为可视域图保存路径、遥感图路径、dem路径
    :param ascension_results_path:
    :return:
    '''

    # 结果保存路径
    pic_path = r"D:/village_data/DongZuDatas/ascension_pics/"

    if os.path.exists(pic_path) is False:
        os.makedirs(pic_path)

    for file in os.listdir(ascension_results_path):
        # 打开村落的遥感图准备绘制
        filename = file.split('dem')[0]
        remote_data = np.array(Image.open(os.path.join(remotes_path,filename + 'remote.png')))
        dem_data = np.array(Image.open(os.path.join(dems_path,filename + 'dem.tif')))

        X_rules, Y_rules = remote_data.shape[0] / dem_data.shape[0], remote_data.shape[1] / dem_data.shape[1]
        ascensionpoints_list_path = os.path.join(ascension_results_path, "".join((file,'/',str(60),'/',"ascensionpoint.txt")))

        # 获得对应村子的登临点集合
        ascensionpoints_list = np.loadtxt(ascensionpoints_list_path)

        # 绘制100个登临点
        for i in range(100):
            point = ascensionpoints_list[i]
#             point[0]为列，point[1]为行

            cv.circle(remote_data,(int(point[0] * X_rules),int(point[1]*Y_rules)),20,(255,0,0),10)

        cv.imwrite(pic_path + filename + "ascensionpoints.png", remote_data)


def batch_processing_exposivepoints(dems_path):
    from ascensionpoint_generate.interfaces.interface_visionpoint import interface_cal_exposivepoint
    # 读取遥感图中每一个文件
    for file in os.listdir(dems_path):
        dem_path = os.path.join(dems_path,file)
        interface_cal_exposivepoint(dem_path, 60, 50)


def batch_processing_draw_exposivepoints_pics(dems_path,remotes_path):
    # 结果保存路径
    pic_path = r"D:/village_data/DongZuDatas/exposive_pics/"

    if os.path.exists(pic_path) is False:
        os.makedirs(pic_path)

    for file in os.listdir(remotes_path):
        file_name = file.split("remote")[0]

        remote_data = np.array(Image.open(os.path.join(remotes_path, file_name + 'remote.png')))
        dem_data = np.array(Image.open(os.path.join(dems_path, file_name + 'dem.tif')))

        X_rules, Y_rules = remote_data.shape[0] / dem_data.shape[0], remote_data.shape[1] / dem_data.shape[1]

        file_path = os.path.join(VIEWSHED_PATH, "".join((file_name,"dem","/","60","/","exposivepoints.txt")))

        exposivepoints = np.loadtxt(file_path)

        for i in range(len(exposivepoints)):
            exposivepoint = exposivepoints[i]
            cv.circle(remote_data,(int(exposivepoint[0] * X_rules),int(exposivepoint[1]*Y_rules)),20,(0,255,0),10)

        cv.imwrite(pic_path + file_name + "exposivepoints.png", remote_data)

if __name__ == "__main__":
    batch_images_path = r"D:\village_data\DongZuDatas"
    dems_images_path = r"D:\village_data\DongZuDatas\dems"
    remotes_images_path = r"D:\village_data\DongZuDatas\remotes"
    # batch_process_village_edge(batch_images_path)

    # batch_processing_village_accessableArea(batch_images_path)

    # batch_processing_ascensionpoints(batch_images_path)

    # batch_processing_drawpics(r"D:\projects\webGisProject_backEnd20220903\static\result\denglin",
    #                           r"D:\village_data\DongZuDatas\dems",
    #                           r"D:\village_data\DongZuDatas\remotes")

    # batch_processing_exposivepoints(dems_images_path)

    batch_processing_draw_exposivepoints_pics(dems_images_path,remotes_images_path)