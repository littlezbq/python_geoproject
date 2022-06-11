import cv2
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from cal_ascensionpoints.cdld import CalculateVisionPoint, Tools
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

    # 将可视域文件与遥感图进行叠加
    # ascensionpoint_area = np.array(Image.open(remoteDomin).convert('RGB'))

    # dem与遥感坐标转换
    # dem = Image.open(dem_path)
    # remote = Image.open(remoteDomin)

    # X_rules, Y_rules = remote.size[0] / dem.size[0], remote.size[1] / dem.size[1]

    # 画出登临点群
    # for i in range(denglin_num):
    #     denglinPoint_resultList_tmpX = ascensionpoint_resultList[i][0] * X_rules
    #     denglinPoint_resultList_tmpY = ascensionpoint_resultList[i][1] * Y_rules
    #     cv.circle(ascensionpoint_area, (denglinPoint_resultList_tmpX.astype(int),
    #                                  denglinPoint_resultList_tmpY.astype(int)), 30, (80, 210, 130), 4)

    # 前denglin_num个登临点列表
    ascensionpoint_numlist_relative = [(float(ascensionpoint_resultlist[i][0]), float(ascensionpoint_resultlist[i][1]))
                                       for i in range(denglin_num)]

    # 保存前ascensionpoint_num个登临点的可视域列表
    ascensionpoint_viewshedpathlist = ["".join(
        (ascensionpoint_basepath, '/', str(int(point[0])), '_', str(int(point[1])), 'ascensionpoint_viewshed.png')) for
        point in ascensionpoint_numlist_relative]

    # 转换坐标为经纬度
    tl = Tools()
    ascensionpoint_numlist_absolute = []
    # dem_array = np.array(dem)
    for i in range(len(ascensionpoint_numlist_relative)):
        transpoint, _ = tl.coordinateTransform(dem_path, ascensionpoint_numlist_relative[i][0],
                                               ascensionpoint_numlist_relative[i][1])
        # ascensionpoint_numlist_absolute.append([transpoint,float(dem_array[int(ascensionpoint_numlist_relative[i][0]), int(ascensionpoint_numlist_relative[i][1])])])
        ascensionpoint_numlist_absolute.append(transpoint)

    # 在图上标注出可视域的大小
    # text = str(denglinPoints[0][2].astype(int))
    # cv.putText(denglinPointArea, text, (denglinPoints[0][0].astype(int)-10, denglinPoints[0][1].astype(int)-10),
    #            cv.FONT_ITALIC, 0.3, (100, 200, 200), 1)

    # cv.imwrite(ascensionpoint_areapath, ascensionpoint_area)
    print("done")

    # 将可视域文件与遥感图进行叠加
    # denglinPointArea1 = np.array(Image.open(denglinPointArea).convert('RGB'))
    # for i in range(100):
    #     denglinViewshed = os.path.join(ascensionPoint_basepath, 
    #                                 "".join((str(denglinPoint_resultList[i][0].astype(int)), '_', str(denglinPoint_resultList[i][1].astype(int)), "denglinPointViewShed.png")))
    #     combineViewShedAndRemote = os.path.join(ascensionPoint_basepath,
    #                                 "".join((str(denglinPoint_resultList[i][0].astype(int)), '_', str(denglinPoint_resultList[i][1].astype(int)), 'combineViewShedRemote.png')))

    #     denglin_viewshed = np.array(Image.open(denglinViewshed).convert('RGB'))

    # 改变颜色
    # combine_viewShed_remote = Image.blend(denglinPointArea1,denglin_viewshed,0.15)
    # denglin_viewshed2 = np.resize(denglin_viewshed,denglinPointArea1.shape)
    # combine_viewShed_remote = cv.addWeighted(denglinPointArea1, 0.85, denglin_viewshed2, 0.15, 0.2)
    # combine_viewShed_remote = np.array(Image.blend(denglinPointArea1, denglin_viewshed2, 0.3))
    # cv.imwrite(combineViewShedAndRemote,combine_viewShed_remote)

    # 返回需要观测的登临点列表，以及登临点计算结果所在的根目录
    return ascensionpoint_numlist_absolute, ascensionpoint_viewshedpathlist


def interface_cal_exposivepoint(dem_path, cur_dis, exposive_num):
    """"

    """

    # 读取可达域内点的列表
    accessarea_point_path = os.path.join(param.VIEWSHED_PATH,
                                         "".join((os.path.basename(dem_path).split('.')[0], "/", str(cur_dis))))

    x_y = np.loadtxt(accessarea_point_path + "/accessdomainpoint_index.txt", dtype=int)

    # 计算出可达域内可视域叠加情况，返回可视域叠加图
    cal_as = CalculateVisionPoint(x_y, dem_path, cur_dis)
    result_baolu = cal_as.calculate_exposivepoint(cur_dis)

    result_exposive_combine = Image.fromarray(result_baolu).convert('RGB')
    result_exposive_combine.save(os.path.join(accessarea_point_path, "result_exposive_combine.png"))

    # Get the exposive_point corrd in the combined viewshed graph
    exposivepoints_numlist_relative = np.argwhere(result_baolu == 255)

    tl = Tools()
    exposivepoint_numlist_absolute = []
    # Translate relative corrds to absolute corrds(longtitude & latitude)
    for tmp_num in range(exposive_num):
        exposive_point = exposivepoints_numlist_relative[tmp_num]
        transpoint, _ = tl.coordinateTransform(dem_path, exposive_point[0], exposive_point[1])
        exposivepoint_numlist_absolute.append(transpoint)

    # Get which point can "see" the exposive_point
    viewshed_path = os.path.join(param.VIEWSHED_PATH,
                                 os.path.join(os.path.basename(dem_path).split(".")[0], str(cur_dis)))
    for num in range(exposive_num):
        # Create an array to display how many points can see the exposive_point
        exposive_parent_array = np.ones_like(np.array(Image.open(dem_path)))

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

                    exposive_parent_array[tuple(map(int, parent_name))] = 255

        np.savetxt(os.path.join(viewshed_path, "exposivepoint_parent" + str(num) + ".txt"), np.array(tmp_list),
                   fmt="%s")
        cv2.imwrite(os.path.join(viewshed_path, "exposivepoint_parent_array" + str(num) + ".png"),
                    exposive_parent_array)

    return exposivepoint_numlist_absolute
