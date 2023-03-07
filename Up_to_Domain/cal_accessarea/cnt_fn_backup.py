from functools import partial
import numpy as np
from PIL import Image
import matplotlib
import time

from multiprocessing import Pool

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2 as cv
from osgeo import gdal
from pathlib import Path
import sys
from Up_to_Domain.graph.initialize_graph import DEM_Graph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.params import CORE_NUM


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class CountSlope(object):
    """
    计算DEM图像的坡度和坡向
    """

    def __init__(self):
        pass

    def read_file(self, file_path):
        """
        读取DEM文件
        :param file_path: DEM文件路径
        :return ndarray格式的DEM数据
        """
        dataset = gdal.Open(file_path)
        im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
        im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
        # im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64) # 将数据写成数组，对应栅格矩阵
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
        return im_data

    def add_round(self, img_array):
        """
        给DEM图像周围加一圈0值
        :param img_array: ndarray格式的DEM数据
        :return: 周围加0值后的DEM图像
        """
        ny, nx = img_array.shape  # ny:行数，nx:列数
        zbc = np.zeros((ny + 2, nx + 2))
        zbc[1:-1, 1:-1] = img_array
        # 四边
        zbc[0, 1:-1] = img_array[0, :]
        zbc[-1, 1:-1] = img_array[-1, :]
        zbc[1:-1, 0] = img_array[:, 0]
        zbc[1:-1, -1] = img_array[:, -1]
        # 角点
        zbc[0, 0] = img_array[0, 0]
        zbc[0, -1] = img_array[0, -1]
        zbc[-1, 0] = img_array[-1, 0]
        zbc[-1, -1] = img_array[-1, 0]
        return zbc

    def cnt_dxdy(self, img_array, sizex=30, sizey=30):
        """
        计算xy方向的梯度
        :param img_array: ndarray格式的DEM图像
        :param sizex: DEM图像x方向格网大小
        :param sizey: DEM图像y方向格网大小
        :return:
        dx: x方向的梯度
        dy: y方向的梯度
        """
        zbc = self.add_round(img_array)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / sizex / 2 / 1000
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / sizey / 2 / 1000

        # dx = dx[1:-1, 1:-1]
        # dy = dy[1:-1, 1:-1]
        return dx, dy

    def AddRound(self, image):
        """
        在图像的周围填充像素，填充值与边缘像素相同

        Parameters
        ----------
        image: ndarray

        Return
        ------
        addrounded_image: ndarray
        """
        ny, nx = image.shape  # ny:行数，nx:列数
        result = np.zeros((ny + 2, nx + 2))
        result[1:-1, 1:-1] = image
        # 四边
        result[0, 1:-1] = image[0, :]
        result[-1, 1:-1] = image[-1, :]
        result[1:-1, 0] = image[:, 0]
        result[1:-1, -1] = image[:, -1]
        # 角点
        result[0, 0] = image[0, 0]
        result[0, -1] = image[0, -1]
        result[-1, 0] = image[-1, 0]
        result[-1, -1] = image[-1, 0]
        return result

    def cal_slope(self, image, grad_we, grad_sn):
        """
        计算一张图片的坡度，使用三阶不带权差分法计算坡度

        Parameters
        ----------
        img_array: ndarray
        grad_we: float
            dem格网宽度，每像素代表的距离（单位：米)
        grad_sn: float
            dem格网高度，每像素代表的距离（单位：米）

        Return
        ------
        slope: ndarray
            坡度图，单位是弧度
        """

        # 这是三阶反距离平方权差分
        # kernal_we = np.array([[1, 0, -1],
        #                       [2, 0, -2],
        #                       [1, 0, -1]])
        # kernal_sn = np.array([[-1, -2, -1],
        #                       [0, 0, 0],
        #                       [1, 2, 1]])

        kernal_we = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])
        kernal_sn = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]])

        img = self.AddRound(image)
        slope_we = cv.filter2D(img, -1, kernal_we)
        slope_sn = cv.filter2D(img, -1, kernal_sn)
        # slope_we = slope_we[1:-1, 1:-1] / 8 / grad_we
        # slope_sn = slope_sn[1:-1, 1:-1] / 8 / grad_sn

        slope_we = slope_we[1:-1, 1:-1] / 6 / grad_we
        slope_sn = slope_sn[1:-1, 1:-1] / 6 / grad_sn

        # slope = (np.arctan(np.sqrt(slope_we * slope_we + slope_sn * slope_sn))) * 57.29578
        slope = (np.arctan(np.sqrt(slope_we * slope_we + slope_sn * slope_sn)))
        return slope

    def CacSlopAsp(self, dx, dy):
        """
        计算坡度和坡向
        :param dx: x方向的梯度
        :param dy: y方向的梯度
        :return:
        slope: 坡度
        spect: 坡向
        """
        slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578  # 转换成°
        slope = slope[1:-1, 1:-1]
        # 坡向
        # aspect = np.zeros([dx.shape[0], dx.shape[1]]).astype(np.float32)
        # for i in range(dx.shape[0]):
        #     for j in range(dx.shape[1]):
        #         x = float(dx[i, j])
        #         y = float(dy[i, j])
        #         if (x == 0.) & (y == 0.):
        #             aspect[i, j] = -1
        #         elif x == 0.:
        #             if y > 0.:
        #                 aspect[i, j] = 0.
        #             else:
        #                 aspect[i, j] = 180.
        #         elif y == 0.:
        #             if x > 0:
        #                 aspect[i, j] = 90.
        #             else:
        #                 aspect[i, j] = 270.
        #         else:
        #             aspect[i, j] = float(math.atan(y / x)) * 57.29578
        #             if aspect[i, j] < 0.:
        #                 aspect[i, j] = 90. - aspect[i, j]
        #             elif aspect[i, j] > 90.:
        #                 aspect[i, j] = 450. - aspect[i, j]
        #             else:
        #                 aspect[i, j] = 90. - aspect[i, j]
        # return slope, aspect
        return slope


class CountUptoDomain(CountSlope):
    """
    计算可达域
    """

    def __init__(self, file_path, point, demAxis):
        """
        :param file_path: DEM文件路径
        :param point: 出发点
        :param demAxis: dem数据精度，x和y相同
        """
        super(CountUptoDomain, self).__init__()
        self.DEM, self.im_geotrans = self.read_file(file_path)
        self.dem_list = np.argwhere(self.DEM)
        self.point = point
        self.demAxis = demAxis

        self.distance_matrix = np.zeros_like(self.DEM)
        self.relative_velocity_matrix = np.zeros_like(self.DEM)
        self.final_velocity_matrix = np.zeros_like(self.DEM)
        self.time_matrix = np.zeros_like(self.DEM)
        self.big_path = []
        self.slope = np.zeros_like(self.DEM)
        self.DEM_Graph = DEM_Graph(self.DEM)

    def read_file(self, file_path):
        """
        读取DEM文件
        :param file_path: DEM文件路径
        :return ndarray格式的DEM数据
        """
        # img = Image.open(file_path)
        # img_array = np.array(img)

        dataset = gdal.Open(file_path)
        im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
        im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
        return im_data, im_geotrans

    def get_sptial_distance(self, currentPoint, tmpPoint):
        space_distance = np.sqrt(
            np.square(np.abs(currentPoint.x - tmpPoint.x) * self.demAxis) +
            np.square(np.abs(currentPoint.y - tmpPoint.y) * self.demAxis)
        )

        sptial_distance = np.sqrt(
            np.square(space_distance) +
            np.square(
                np.abs(self.DEM[int(currentPoint.x)][int(currentPoint.y)] - self.DEM[int(tmpPoint.x)][int(tmpPoint.y)]))
        )

        return sptial_distance

    def get_paths(self, currentPoint, tmpPoint):
        """
        计算两点之间的距离
        :param currentPoint: 当前的中心点
        :param tmpPoint: 目标点
        :return:
        paths: 当前点到目标点走过的路径
        """
        paths = []
        i, j = int(currentPoint.x), int(currentPoint.y)

        # paths.append([x,y])
        # 第4象限
        if (tmpPoint.x >= currentPoint.x) and (tmpPoint.y >= currentPoint.y):
            for ii in range(int(abs(tmpPoint.x - currentPoint.x) + abs(tmpPoint.y - currentPoint.y))):
                if tmpPoint.x > i:
                    paths.append([i, j])
                    i += 1
                if tmpPoint.y > j:
                    paths.append([i, j])
                    j += 1

                if tmpPoint.x == i and tmpPoint.y == j:
                    paths.append([i, j])
                    break

        # 第3象限
        elif (tmpPoint.x <= currentPoint.x) and (tmpPoint.y >= currentPoint.y):
            for ii in range(int(abs(currentPoint.x - tmpPoint.x) + abs(currentPoint.y - tmpPoint.y))):
                if tmpPoint.x < i:
                    paths.append([i, j])
                    i -= 1
                if tmpPoint.y > j:
                    paths.append([i, j])
                    j += 1

                if tmpPoint.x == i and tmpPoint.y == j:
                    paths.append([i, j])
                    break
        # 第2象限
        elif (tmpPoint.x <= currentPoint.x) and (tmpPoint.y <= currentPoint.y):
            for ii in range(int(abs(currentPoint.x - tmpPoint.x) + abs(currentPoint.y - tmpPoint.y))):
                if tmpPoint.x < i:
                    paths.append([i, j])
                    i -= 1
                if tmpPoint.y < j:
                    paths.append([i, j])
                    j -= 1

                if tmpPoint.x == i and tmpPoint.y == j:
                    paths.append([i, j])
                    break
        # 第1象限
        elif (tmpPoint.x >= currentPoint.x) and (tmpPoint.y <= currentPoint.y):
            for ii in range(int(abs(currentPoint.x - tmpPoint.x) + abs(currentPoint.y - tmpPoint.y))):
                if tmpPoint.x > i:
                    paths.append([i, j])
                    i += 1
                if tmpPoint.y < j:
                    paths.append([i, j])
                    j -= 1

                if tmpPoint.x == i and tmpPoint.y == j:
                    paths.append([i, j])
                    break

        return paths

    def get_path_new(self, srcNode, dstNode):
        """
        Using DEM_Graph methods
        :param srcNode: sourceNode
        :param dstNode: destinationNode
        :return: node list between src and dst
        """

        return self.DEM_Graph.dijkstra_path(srcNode,dstNode)


    def get_slope(self):
        # DEM = self.add_round(self.DEM)
        # dx, dy = self.cnt_dxdy(DEM, self.xaxis, self.yaxis)
        # slope = self.CacSlopAsp(dx, dy)
        xaxis = self.demAxis
        yaxis = self.demAxis
        slope = self.cal_slope(self.DEM, xaxis, yaxis)
        return slope

    def map_all_paths(self, tpoint):
        '''
            Get one path of a point to centerpoint
        '''
        print("Mapping every point in dem....")
        path = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))
        return path

    def get_all_paths(self):
        '''
            Generaing paths of all points to a centerpoint
        '''

        st = time.time()
        print("Generaing all paths to centerpoint......")
        with Pool(processes=CORE_NUM) as p:
            big_path = p.map(func=self.map_all_paths, iterable=self.dem_list)
        ed = time.time()
        print("Done!, using{:.2f} seconds".format((ed - st)))

        self.big_path = big_path

    def map_distance_matrix(self, tpoint):
        # print("Mapping every point in dem, getting one target point distance to centerpoint")
        sumdistance = 0
        # 获取当前点与中心点之间所有经过的点的集合
        paths = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))
        for index in range(len(paths) - 1):
            tmp_point_sptial_distance = self.get_sptial_distance(Point(paths[index][0], paths[index][1]),
                                                                 Point(paths[index + 1][0], paths[index + 1][1])) * 2
            sumdistance = sumdistance + tmp_point_sptial_distance

        return sumdistance

    def map_distance_matrix_new(self, tpoint):
        """

        :param tpoint:
        :return: centerpoint to tpoint distance
        """
        sumdistance = 0
        # 封装节点
        index_node_src = self.point.x * self.DEM.shape[1] + self.point.y
        index_node_dst = tpoint[0] * self.DEM.shape[1] + tpoint[1]
        paths = self.get_path_new(index_node_src,index_node_dst)

        for index in range(len(paths) - 1):
            x_former = paths[index] // self.DEM.shape[1]
            y_former = paths[index] - self.DEM.shape[1] * x_former

            x_latter = paths[index + 1] // self.DEM.shape[1]
            y_latter = paths[index + 1] - self.DEM.shape[1] * x_latter

            former_point = Point(x_former, y_former)
            latter_point = Point(x_latter,y_latter)
            tmp_point_sptial_distance = self.get_sptial_distance(former_point, latter_point) * 2
            sumdistance = sumdistance + tmp_point_sptial_distance

        return sumdistance


    def map_distance_matrix2(self, tpoint):
        '''
            直接计算目标点到中心点的欧氏距离, no use
        '''
        tmp_point_sptial_distance = self.get_sptial_distance(self.point, Point(tpoint[0], tpoint[1]))

        return tmp_point_sptial_distance

    def get_distance_matrix(self):
        # Create a multiprocessing pool
        st = time.time()
        print("Calculating distance matrix......")
        with Pool(processes=CORE_NUM) as p:
            # Using multiprocessing to calculate
            distance_list = p.map(func=self.map_distance_matrix_new, iterable=self.dem_list)
            # distance_list = p.map(func=self.map_distance_matrix, iterable=self.dem_list)

        ed = time.time()
        print("Done!, using {:.2f} seconds".format((ed - st)))
        self.distance_matrix = np.array(distance_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

        return self.distance_matrix

    def map_relative_velocity_matrix(self, slope, tpoint):
        '''
            Calculate each point in DEM
        '''
        # print("Mapping every point in dem, getting one point relative velocity....")
        tmp = np.cos((slope[tpoint[0], tpoint[1]] + 0.001))
        if slope[tpoint[0], tpoint[1]] > 0:
            velocity = 4000 * np.abs(tmp)
        elif slope[tpoint[0], tpoint[1]] < 0:
            velocity = 4000 / np.abs(tmp)
        else:
            velocity = 4000

        return velocity

    def get_relative_velocity_matrix(self):
        '''
            Getting relative velocity matrix
        '''
        st = time.time()
        print("Calculating relative velocity...")

        with Pool(processes=CORE_NUM) as p:
            self.slope = self.get_slope()
            funcs = partial(self.map_relative_velocity_matrix, self.slope)
            velocity_list = p.map(func=funcs, iterable=self.dem_list)

        ed = time.time()
        print("Done!,using{:.2f} seconds".format((ed - st)))

        self.relative_velocity_matrix = np.array(velocity_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

    def map_final_velocity_matrix(self, tpoint):
        # Caculate the final speed by 加权平均 relative speed in paths

        # Get the target point path index of big_path
        # print("Getting target path to centerpoint...")
        paths = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))
        relative_velocity_list = []

        # print("Mapping every point in dem, getting one point final velocity......")
        for point in paths:
            relative_velocity_list.append(self.relative_velocity_matrix[point[0], point[1]])
        sum = np.sum(relative_velocity_list)

        if sum != 0:
            weights = np.divide(relative_velocity_list, sum)
            final_velocity = np.average(relative_velocity_list, weights=weights)

        else:
            final_velocity = 0

        return final_velocity

    def get_final_velocity_matrix(self):
        '''
            Calculate final velocity
        '''

        # Get relative velocity matrix first
        self.get_relative_velocity_matrix()

        # Get paths of all points to centerpoint
        # self.get_all_paths()

        st = time.time()
        # print("Calculating final velocity matrix......")

        with Pool(processes=CORE_NUM) as p:
            final_velocity_list = p.map(func=self.map_final_velocity_matrix, iterable=self.dem_list)

        ed = time.time()
        print("Done!, using {} seconds".format((ed - st)))
        self.final_velocity_matrix = np.array(final_velocity_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

        return self.final_velocity_matrix

    def map_get_time_matrix(self, tpoint):
        """
        time=paths/velocity
        :param distance: 距离
        :param velocity: 速度
        :param mode_D: 距离计算模式，分为街区距离和欧式距离
        :param mode_V: 速度计算模式，暂未定
        :return: time: 时间
        """

        # Get the path between center point and each point on dem
        paths = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))

        total_time = 0
        # During the path calculate distance and velocity between each two point
        for index in range(len(paths) - 1):
            tmp_point_sptial_distance = self.get_sptial_distance(Point(paths[index][0], paths[index][1]),
                                                                 Point(paths[index + 1][0],
                                                                       paths[index + 1][1])) * 1.2  # 修正系数
            tmp_point_velocity = (self.relative_velocity_matrix[paths[index][0], paths[index][0]] +
                                  self.relative_velocity_matrix[paths[index + 1][0], paths[index + 1][0]]) / 2
            tmp_point_costtime = tmp_point_sptial_distance / tmp_point_velocity

            total_time += tmp_point_costtime

        return total_time

    def get_final_time_matrix(self):

        #         First calculate the slope and relative_velocity_matrix
        self.get_relative_velocity_matrix()

        st = time.time()
        with Pool(processes=CORE_NUM) as p:
            final_time_list = p.map(func=self.map_get_time_matrix, iterable=self.dem_list)

        ed = time.time()
        print("Done!, using {} seconds".format((ed - st)))
        self.time_matrix = np.array(final_time_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

        return self.time_matrix

    def get_time_matrix_pre(self):
        self.get_distance_matrix()
        self.get_final_velocity_matrix()
        self.time_matrix = np.divide(self.distance_matrix, self.final_velocity_matrix)

        return self.time_matrix


def work_1(file_path):
    '''
        测试1：距离矩阵计算方法为，先按照街区距离计算模式得出目标点距离中心点之间经过的点构成的集合（包含了途经其余点的坐标），接着计算该点集中两两个点间的空间欧式距离，累加得到中心点到该目标点的距离，
    循环计算dem中各点到中心点的距离，构成距离矩阵。
    '''
    import os
    save_dir = './test_accessDomain/mode_0_0_new'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    data = Image.open(file_path)
    point = Point(117, 107)
    demaxis = 8.5
    a = CountUptoDomain(file_path=file_path,
                        point=point,
                        demAxis=demaxis
                        )

    time_matrix = a.get_time_matrix_pre()

    et = time.time()
    print(et - st)

    x_plt = np.arange(0, time_matrix.shape[1], 1)
    y_plt = np.arange(0, time_matrix.shape[0], 1)
    # # #
    # fig1, ax1 = plt.subplots()
    # ax1.invert_yaxis()  # y轴反向
    # plt.contourf(x_plt, y_plt, a.distance_matrix, 25)
    # plt.colorbar()
    # # ax1.imshow(distance_matrix)
    # plt.savefig(os.path.join(save_dir, 'distance_matrix.png'), bbox_inches="tight", pad_inches=0.0)
    # #
    # fig2, ax2 = plt.subplots()
    # ax2.invert_yaxis()  # y轴反向
    # plt.contourf(x_plt, y_plt, a.final_velocity_matrix, 25)
    # plt.colorbar()
    # plt.savefig(os.path.join(save_dir, 'velocity_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    fig3, ax3 = plt.subplots()
    ax3.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, time_matrix, 25)
    plt.colorbar()
    # ax3.imshow(time_matrix)
    plt.savefig(os.path.join(save_dir, 'time_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    kedayu = time_matrix < (15 / 60)
    result = Image.fromarray(kedayu).convert('RGB')

    result.save(os.path.join(save_dir, 'kedayutest.png'))


def work_2(file_path):
    '''
        测试2：距离矩阵计算方法为，直接计算目标点到中心点的空间欧式距离，循环计算dem中其余各点到中心点的距离，构成距离矩阵。
    '''
    import os
    save_dir = './test_accessDomain/mode_0_1'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    data = Image.open(file_path)
    point = Point(data.size[0] / 2, data.size[1] / 2)
    xaxis, yaxis = 10, 10
    a = CountUptoDomain(file_path=file_path,
                        point=point,
                        xaxis=xaxis,
                        yaxis=yaxis)

    time_matrix = a.get_time_matrix(mode_D=1, mode_V=0)

    x_plt = np.arange(0, time_matrix.shape[1], 1)
    y_plt = np.arange(0, time_matrix.shape[0], 1)
    # # #
    # fig1, ax1 = plt.subplots()
    # ax1.invert_yaxis()  # y轴反向
    # plt.contourf(x_plt, y_plt, a.distance_matrix, 25)
    # plt.colorbar()
    # # ax1.imshow(distance_matrix)
    # plt.savefig(os.path.join(save_dir, 'distance_matrix.png'), bbox_inches="tight", pad_inches=0.0)
    # #
    fig1, ax1 = plt.subplots()
    ax1.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, a.distance_matrix, 25)
    plt.colorbar()
    # ax1.imshow(distance_matrix)
    plt.savefig(os.path.join(save_dir, 'distance_matrix.png'), bbox_inches="tight", pad_inches=0.0)
    #
    fig2, ax2 = plt.subplots()
    ax2.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, a.velocity_matrix, 25)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'velocity_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    fig3, ax3 = plt.subplots()
    ax3.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, time_matrix, 25)
    plt.colorbar()
    # ax3.imshow(time_matrix)
    plt.savefig(os.path.join(save_dir, 'time_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    kedayu = time_matrix < (15 / 60)
    result = Image.fromarray(kedayu).convert('RGB')

    result.save(os.path.join(save_dir, 'kedayutest.png'))


if __name__ == '__main__':
    import time

    st = time.time()

    file_path = r'D:\projects\webGisProject_backEnd20220903\static\datasets\SX3_005_QMC.tif'
    work_1(file_path)
    end = time.time()
    print(end - st)
