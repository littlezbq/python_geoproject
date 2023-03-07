from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ascensionpoint_generate.cal_ascensionpoints.cdld import CalculateVisionPoint
from commonutils.tools import Tools
import numpy as np
import os
import config.params as param


def interface_cal_ascensionpoint(dem_path, denglin_num, time_limit, divideSpace, demAxis):
    """
    计算可达域内点的可视域并渲染至可达域上
    :param dem:
    :param remoteDomin:
    :return:
    """

    # 设置登临点算法计算保存的程序基路径
    ascensionpoint_basepath = os.path.join(param.ASCENSIONLIST_SAVEPATH,
                                           os.path.join(os.path.basename(dem_path).split('.')[0], str(time_limit)))

    # 设置可视域计算结果保存的基路径
    viewshedmiddle_basepath = os.path.join(param.VIEWSHED_PATH,
                                           os.path.join(os.path.basename(dem_path).split('.')[0], str(time_limit)))

    # ascensionpoint_areapath = os.path.join(ascensionpoint_basepath,str(denglin_num) + '_' + 'denglinPointArea.png')
    # combineViewShedAndRemote = os.path.join(ascensionpoint_basepath,'combineViewShedRemote.png')

    if Path(ascensionpoint_basepath).exists() is False:
        os.makedirs(ascensionpoint_basepath)

    # 读取可达域内点的列表
    x_y = np.loadtxt(os.path.join(viewshedmiddle_basepath, "accessdomainpoint_index.txt"), dtype=int)

    # 计算登临点,返回登临点列表
    cal_as = CalculateVisionPoint(x_y, dem_path, time_limit)
    ascensionpoint_resultlist = cal_as.calculate_ascensionpoint_mp(divideSpace, demAxis)

    # 前denglin_num个登临点列表
    # 设置
    ascensionpoint_numlist_relative = [(float(ascensionpoint_resultlist[i][0]), float(ascensionpoint_resultlist[i][1]))
                                       for i in range(denglin_num)]

    # 将登临点划分到不同的山头
    # mountainareas = interface_getmountainarea(dem_path)

    # mountainarea_ascensionpoint_list = []
    # for ascensionpoint in ascensionpoint_numlist_relative:
    #     if ascensionpoint[0]

    # 保存前ascensionpoint_num个登临点的可视域列表
    ascensionpoint_viewshedpathlist = ["".join(
        (ascensionpoint_basepath, '/', str(int(point[0])), '_', str(int(point[1])), 'ascensionpoint_viewshed.png')) for
        point in ascensionpoint_numlist_relative]

    # 转换坐标为经纬度

    ascensionpoint_numlist_absolute = []
    # dem_array = np.array(dem)
    for i in range(len(ascensionpoint_numlist_relative)):
        transpoint, _ = Tools.coordinateTransformOfPoint(dem_path, ascensionpoint_numlist_relative[i][0],
                                                         ascensionpoint_numlist_relative[i][1])
        ascensionpoint_numlist_absolute.append(transpoint)

    # 返回需要观测的登临点列表，以及登临点计算结果所在的根目录
    return ascensionpoint_numlist_absolute, ascensionpoint_viewshedpathlist


def interface_cal_exposivepoint(dem_path, cur_dis, exposive_num):
    """"
        input:
            path of the target dem data, timelimit of the accessdomain and number of exposive
        result:
            parent points(.txt) of exposive point and parent array(.png) of exposive point
        return:
            path of parent points(.txt) and parent array(.png)
    """

    # 读取可达域内点的列表
    accessarea_point_path = os.path.join(param.VIEWSHED_PATH,
                                         "".join((os.path.basename(dem_path).split('.')[0], "/", str(cur_dis))))

    x_y = np.loadtxt(accessarea_point_path + "/accessdomainpoint_index.txt", dtype=int)

    # 计算出可达域内可视域叠加情况，返回可视域叠加图
    cal_as = CalculateVisionPoint(x_y, dem_path, cur_dis)
    result_exposive,result_exposive_new = cal_as.calculate_exposivepoint(cur_dis)

    result_exposive_combine = Image.fromarray(result_exposive_new).convert('RGB')
    result_exposive_combine.save(os.path.join(accessarea_point_path, "result_exposive_combine.png"))

    # Get the exposive_point corrd in the combined viewshed graph
    # exposivepoints_numlist_relative = np.argwhere(result_exposive == 255)

    # 直接对叠加矩阵元素值排序
    _, exposivepoints_numlist_relative = wybb(result_exposive)

    # exposivepoints_numlist_relative = [
    #     [55,101],
    #     [72,116],
    #     [130,53],
    #     [103,43],
    #     [81,136]
    # ]

    exposivepoint_numlist_absolute = []
    # Translate relative corrds to absolute corrds(longtitude & latitude)
    for tmp_num in range(exposive_num):
        exposive_point = exposivepoints_numlist_relative[tmp_num]
        transpoint, _ = Tools.coordinateTransformOfPoint(dem_path, exposive_point[0], exposive_point[1])
        exposivepoint_numlist_absolute.append(transpoint)

    # Get which point can "see" the exposive_point
    viewshed_path = os.path.join(param.VIEWSHED_PATH,
                                 os.path.join(os.path.basename(dem_path).split(".")[0], str(cur_dis)))
    for num in range(exposive_num):
        # Create an array to display how many points can see the exposive_point
        # exposive_parent_array = np.ones_like(np.array(Image.open(dem_path)))

        # Get current exposive_point from exposivepoints_numlist_relative
        tmp_list = []
        exposive_point = exposivepoints_numlist_relative[num]

        for file in os.listdir(viewshed_path):
            if file.endswith("tif"):
                viewshed_data = np.array(Image.open(os.path.join(viewshed_path, file)))
                '''
                    Find if the exposive_point can be seen in viewshed, then the visionpoint which generates this 
                    viewshed is the parent point of this exposive_point
                '''

                if viewshed_data[exposive_point[0], exposive_point[1]] == 255:
                    parent_name = file.split('.tif')[0].split('_')
                    tmp_list.append(parent_name)

                    # exposive_parent_array[tuple(map(int, parent_name))] = 255

        exposive_result_path = os.path.join(param.ASCENSIONLIST_SAVEPATH,
                                            os.path.join(os.path.basename(dem_path).split(".")[0], str(cur_dis)))
        exposivepoint_parent_path = os.path.join(exposive_result_path, "exposivepoint_parent" + str(num) + ".txt")
        # exposivepoint_array_path = os.path.join(exposive_result_path, "exposivepoint_parent_array" + str(num) + ".png")
        np.savetxt(exposivepoint_parent_path, np.array(tmp_list), fmt="%s")
        # cv2.imwrite(exposivepoint_array_path, exposive_parent_array)

        np.savetxt(os.path.join(accessarea_point_path,"exposivepoints.txt"),exposivepoints_numlist_relative,fmt="%s")

    exposive_result_path = "/api/" + exposive_result_path
    # exposive_result_path = exposive_result_path
    return exposivepoint_numlist_absolute, exposive_result_path

    # np.savetxt("exposiveOfTXS.txt", exposivepoints_numlist_relative, fmt="%s")
    # return exposivepoints_numlist_relative

def wybb(A):
    Bs = []
    As = []

    x = A.reshape(1, -1)
    y = np.argsort(x)
    y = np.argsort(y).reshape(A.shape)

    for i in range(A.shape[0] * A.shape[1]):
        a = np.argwhere(y == i)
        b = A[a[0][0], a[0][1]]
        Bs.append(b)
        As.append((a[0][0], a[0][1]))

    # 倒序排列
    Bs.reverse()
    As.reverse()
    return Bs, As