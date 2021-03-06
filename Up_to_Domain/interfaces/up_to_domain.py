import os
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from pathlib import Path
from Up_to_Domain.cal_accessarea.cnt_fn import CountUptoDomain, Point
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from village_edge_dev.interfaces import edge_detect
import config.params as param
from ascensionpoint_generate.cal_ascensionpoints.cdld import Tools

def interface_drawRawData(dem_path, remote_path):
    """

    :param dem_path: 输入dem数据路径
    :param remote_path: 输入遥感数据路径
    :return: 将其绘制后的图像路径及图像的中心返回，并可视化到前端页面
    """

    rawDem = os.path.join(param.RAWIMG_PATH,
                          os.path.basename(dem_path).split('.')[0] + 'Dem.png')

    rawRemote = os.path.join(param.RAWIMG_PATH,
                             os.path.basename(remote_path).split('.')[0] + 'Remote.png')

    # 路径存在校验
    if os.path.exists(param.RAWIMG_PATH) is False:
        os.makedirs(param.RAWIMG_PATH)

    # 原始dem
    fig1, ax1 = plt.subplots()
    dem = np.array(Image.open(dem_path))

    # 读取dem并获取中心点坐标
    tl = Tools()
    im_data, im_geotrans, im_width, im_height = tl.read_dem(filename=dem_path)
    x_mean, y_mean = im_width / 2.0, im_height / 2.0

    # 转换相对坐标为经纬度
    start_longtitude = im_geotrans[0]
    start_latitude = im_geotrans[3]
    index_longitude = im_geotrans[1]
    index_latitude = im_geotrans[5]
    print(im_geotrans)
    # 转换后的经纬度坐标
    px = start_longtitude + x_mean * index_longitude + y_mean * im_geotrans[2]
    py = start_latitude + x_mean * im_geotrans[4] + y_mean * index_latitude

    # remote
    remote = cv.imread(remote_path)
    cv.imwrite(rawRemote, remote)

    ax1.imshow(dem, cmap="gray")
    dem_size = dem.shape
    remote_size = remote.shape
    plt.axis('off')
    plt.savefig(rawDem, bbox_inches="tight", pad_inches=0.0)
    plt.close()

    # resize一下
    dem2 = Image.open(rawDem)
    dem2 = dem2.resize(dem_size)
    dem2.save(rawDem)

    return rawDem, rawRemote, [px, py]


def interface_uptodomain(dem_path, remote_path, timelimit, demAxis):
    """
    :param dem_path: 输入dem路径
    :param remote_path: 输入遥感图路径
    :param timelimit: 步行时间
    :param xaxis: x轴精度
    :param yaxis: y轴精度
    :return: 村落中心点及可达域的图像


    计算过程保存变量：
        accessdomainpoint_index：可达域内点的坐标
        time_matrix：以村落中心为起点到其余各点的时间花费矩阵
    """

    # eg. ../static/result/reachableArea/reachableImg/SX3_005_QMC
    result_accessdomain_basepath = "".join(
        (os.path.join(param.REACHABLEIMG_PATH, os.path.basename(dem_path).split('.')[0])))

    # eg. ../static/result/reachableArea/reachableImg/SX3_005_QMC/30
    timelimit_accessdomain_path = os.path.join(result_accessdomain_basepath,str(timelimit))

    # eg. ../static/result/reachableArea/reachableImg/SX3_005_QMC/30/accessdomain30.png
    # The Path to locate the accessdomain of paticular timelimit 
    accessdomain_path = os.path.join(timelimit_accessdomain_path, "".join(('accessdomain', str(timelimit), ".png")))

    # eg. ../static/result/reachableArea/reachableImg/SX3_005_QMC/30/accessdomainpoint_index.txt
    # The indexs of the points in accessdomain
    accessdomainpoint_index_path = os.path.join(timelimit_accessdomain_path, "accessdomainpoint_index.txt")

    # ../static/result/reachableArea/reachableImg/SX3_005_QMC/time_matrix.txt
    # Only one dem file has one time_matrix.txt, it has the time_matrix of total points
    time_matrix_path = os.path.join(result_accessdomain_basepath, "time_matrix.txt")

    viewshedmiddle_basepath = os.path.join(param.VIEWSHED_PATH,
                                        os.path.join(os.path.basename(dem_path).split('.')[0],str(timelimit)))


    if os.path.exists(result_accessdomain_basepath) is False:
        os.makedirs(result_accessdomain_basepath)

    if os.path.exists(timelimit_accessdomain_path) is False:
        os.makedirs(timelimit_accessdomain_path)

    # Change the accessdomain from minutes to hour
    timelimit = timelimit / 60.0

    st = time.time()

    # 计算村落中心点
    [x, y] = edge_detect.interface_villageEdgeDect(remote_path)

    ed = time.time()

    print("Detecting edges: ",ed-st)

    # 坐标转换，将从遥感图上计算出得中心点转换到dem上
    dem = Image.open(dem_path)
    remote = Image.open(remote_path)

    X_rules, Y_rules = remote.size[0] / dem.size[0], remote.size[1] / dem.size[1]

    dem_x, dem_y = round(x / X_rules), round(y / Y_rules)

    centerPoint = Point(dem_x, dem_y)

    '''
        If time_matrix has not been calculated, it means the particular village image is first seen by the algorithm,
        so calculate the time_matrix first. Next time only cut the needed accessdomain from the time_matrix using:
            accessdomain = time_matrix < (timelimit)
        noted: timelimit has to be transform to hour
    '''
    st = time.time()
    if os.path.exists(time_matrix_path) is False:
        cnt_uptodomain = CountUptoDomain(dem_path, centerPoint, demAxis)

        # distance_matrix = cnt_uptodomain.get_distance_matrix()
        # velocity_matrix = cnt_uptodomain.get_velocity_matrix()
        time_matrix = cnt_uptodomain.get_time_matrix()

        # 保存可达域内点信息
        Ts_denglinPoint = time_matrix
        # 保存时间矩阵
        np.savetxt(time_matrix_path, time_matrix)

        ed = time.time()
        print("Generating accessdomain:",ed - st)

    else:
        # If has calculated, read the cache directly
        time_matrix = np.loadtxt(time_matrix_path)
        Ts_denglinPoint = time_matrix

        ed = time.time()
        print("Generating accessdomain:",ed - st)

    kedayu_matrix = Ts_denglinPoint < (timelimit)

    # 注意第一维坐标为y，第二维是x
    index = np.argwhere(kedayu_matrix)

    # index_read = np.array([(index[:, 1][i], index[:, 0][i]) for i in range(len(index))])
    # 保存一份到当前路径，另外一份到登临点的中间结果目录下
    if os.path.exists(viewshedmiddle_basepath) is False:
        os.makedirs(viewshedmiddle_basepath)

    np.savetxt(accessdomainpoint_index_path, index, fmt="%d")
    np.savetxt(os.path.join(viewshedmiddle_basepath,"accessdomainpoint_index.txt"), index, fmt="%d")

    # Transform relative coords to longtitude/latitude
    tl = Tools()
    centerpoint, im_geotrans = tl.coordinateTransform(dem_path, dem_x, dem_y)

    # 创建在前端显示的区域范围
    west = im_geotrans[0]
    south = im_geotrans[3] + dem.size[1]*im_geotrans[-1]
    east = im_geotrans[0] + dem.size[0]*im_geotrans[1]
    north = im_geotrans[3]

    # Generate the accessdomain
    kedayu_result = Image.fromarray(kedayu_matrix).convert('RGB')

    kedayu_result.save(accessdomain_path)

    # Return the accessdomain
    return centerpoint, accessdomain_path, [west, south, east, north]