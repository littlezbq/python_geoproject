import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ascensionpoint_generate.interfaces import interface_visionpoint
from Up_to_Domain.interfaces import up_to_domain
import os
from commonutils.tools import Tools
from config.params import VIEWSHED_PATH, ASCENSIONLIST_SAVEPATH
import cv2 as cv


def running_kedayu(remotes_path, dems_path, time_limit, start_points_path, dem_axis):
    start_points = np.loadtxt(start_points_path)
    for i, [remote_path, dem_path] in enumerate(zip(os.listdir(remotes_path), os.listdir(dems_path))):
        # 跑可达域的程序
        dem_path_ = os.path.join(dems_path, dem_path)
        remote_path_ = os.path.join(remotes_path, remote_path)
        start_point = start_points[i]
        up_to_domain.interface_uptodomain(dem_path=dem_path_,
                                          remote_path=remote_path_,
                                          timelimit=time_limit,
                                          demAxis=dem_axis,
                                          start_point=start_point)


def running_denglin_exposive(dems_path, time_limit, dem_axis, ):
    for dem_path in os.listdir(dems_path):
        # 跑登临点的程序
        dem_path_ = os.path.join(dems_path, dem_path)

        interface_visionpoint.interface_cal_ascensionpoint(
            dem_path=dem_path_,
            denglin_num=denglin_num,
            time_limit=time_limit,
            divideSpace=60,
            demAxis=dem_axis
        )

        #         跑暴露点的程序
        interface_visionpoint.interface_cal_exposivepoint(
            dem_path=dem_path_,
            cur_dis=time_limit,
            exposive_num=exposive_num
        )


def drawing_pics(remotes_path, dems_path, time_limit, denglin_num, exposive_num, save_path):
    #     绘制登临点暴露点到遥感大图上
    for remote_path, dem_path in zip(os.listdir(remotes_path), os.listdir(dems_path)):
        dem_path_ = os.path.join(dems_path, dem_path)
        remote_path_ = os.path.join(remotes_path, remote_path)

        #       读取dem和遥感
        dem_data, _, dem_width, dem_height = Tools.read_dem(dem_path_)
        remote_data = np.array(Image.open(remote_path_))

        remote_width, remote_height = remote_data.shape[1], remote_data.shape[0]

        #         计算行和列的尺度缩放
        row_rules, col_rules = remote_height / dem_height, remote_width / dem_width

        del dem_data

        #         读取登临点的文件列表，选择前20个登临点
        denglin_path = os.path.join(os.path.join(ASCENSIONLIST_SAVEPATH, dem_path.split(".tif")[0]), str(time_limit))

        exposive_path = os.path.join(os.path.join(VIEWSHED_PATH, dem_path.split(".tif")[0]), str(time_limit))

        denglin_points = np.loadtxt(os.path.join(denglin_path, "ascensionpoint.txt"))
        exposive_points = np.loadtxt(os.path.join(exposive_path, "exposivepoints.txt"))

        #         将登临点和暴露点绘制到遥感上
        for i in range(denglin_num):
            denglin_point = denglin_points[i]
            cv.circle(remote_data, (int(denglin_point[0] * row_rules), int(denglin_point[1] * col_rules)), 20,
                      (255, 0, 0))

        for i in range(exposive_num):
            exposive_point = exposive_points[i]
            cv.circle(remote_data, (int(exposive_point[0] * row_rules), int(exposive_point[1] * col_rules)), 20,
                      (0, 255, 0))

        cv.imwrite(os.path.join(save_path, remote_path + ".png"), remote_data)


if __name__ == "__main__":

    dems_path = r"G:\projects\datas\ChengGuiData\dem"
    remotes_path = r"G:\projects\datas\ChengGuiData\remote"
    time_limit = 120
    dem_axis = 30
    denglin_num = 20
    exposive_num = 20
    save_path = r"G:\projects\webGisProject_backEnd20220903\static\toChenggui"
    start_points_path = r"G:\projects\datas\ChengGuiData\start_point\start_point.txt"

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    running_kedayu(remotes_path, dems_path, time_limit, start_points_path, dem_axis)
    running_denglin_exposive(dems_path, time_limit, dem_axis)
    drawing_pics(remotes_path, dems_path, time_limit, denglin_num, exposive_num, dem_axis)
