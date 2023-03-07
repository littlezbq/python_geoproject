from functools import partial
import numpy as np
from PIL import Image
import matplotlib
import time
import cProfile
import multiprocessing
from multiprocessing import Pool

import config.params

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2 as cv
from osgeo import gdal
from pathlib import Path
import sys
from Up_to_Domain.graph.initialize_graph import DEM_Graph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.params import CORE_NUM
from Up_to_Domain.cal_accessarea.point import Point
from Up_to_Domain.cal_accessarea.cal_slope import CountSlope


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

        self.distance_matrix = np.zeros_like(self.DEM, dtype=np.float64)
        self.relative_velocity_matrix = np.zeros_like(self.DEM, dtype=np.float64)
        self.final_velocity_matrix = np.zeros_like(self.DEM, dtype=np.float64)
        self.time_matrix = np.zeros_like(self.DEM, dtype=np.float64)
        self.shortest_path = []
        self.slope = np.zeros_like(self.DEM)
        self.DEM_Graph = DEM_Graph(self.DEM, demAxis)

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

    @DeprecationWarning
    def get_sptial_distance(self, currentPoint, tmpPoint):
        """
        Deprecated: This API has moved to DEM_Graph class
        估算计算两格网的空间距离
        :param currentPoint:
        :param tmpPoint:
        :return:
        """
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

    @DeprecationWarning
    def get_paths(self, currentPoint, tmpPoint):
        """
        Deprecated:
        计算两点之间的距离,采取棋盘距离计算
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

    def get_shortest_paths(self, srcNode):
        """
        Using DEM_Graph methods
        :param srcNode: sourceNode
        :return: node list between src and all node in Graph
        """

        self.shortest_path = self.DEM_Graph.shortest_path(srcNode)

    def get_slope(self):
        # DEM = self.add_round(self.DEM)
        # dx, dy = self.cnt_dxdy(DEM, self.xaxis, self.yaxis)
        # slope = self.CacSlopAsp(dx, dy)
        xaxis = self.demAxis
        yaxis = self.demAxis
        slope = self.cal_slope(self.DEM, xaxis, yaxis)
        return slope

    @DeprecationWarning
    def map_all_paths(self, tpoint):
        '''
            Deprecated: Get one path of a point to centerpoint
        '''
        print("Mapping every point in dem....")
        path = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))
        return path

    @DeprecationWarning
    def get_all_paths(self):
        '''
            Deprecated: Generaing paths of all points to a centerpoint
        '''

        st = time.time()
        print("Generaing all paths to centerpoint......")
        with Pool() as p:
            big_path = p.map(func=self.map_all_paths, iterable=self.dem_list)
        ed = time.time()
        print("Done!, using{:.2f} seconds".format((ed - st)))

        self.big_path = big_path

    @DeprecationWarning
    def map_distance_matrix(self, tpoint):
        """
        Deprecated: Using map_distance_matrix_new in this
        :param tpoint:
        :return:
        """
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

        # st = time.time()

        def two_point_distance(index):
            sum = self.DEM_Graph.G[paths[index]][paths[index + 1]]['weight']

            return sum

        sumdistance = 0
        # 封装节点
        index_node_dst = tpoint[0] * self.DEM.shape[1] + tpoint[1]
        try:
            # 如果最短路径中不存在改点，说明该点不可达，将路径置成无限大
            paths = self.shortest_path[index_node_dst]
            sumdistance += sum([two_point_distance(index) for index in range(len(paths) - 1)])
        except:
            sumdistance = float('inf')

        return sumdistance

    @DeprecationWarning
    def map_distance_matrix2(self, tpoint):
        '''
            Deprecated: 直接计算目标点到中心点的欧氏距离, no use
        '''
        tmp_point_sptial_distance = self.get_sptial_distance(self.point, Point(tpoint[0], tpoint[1]))

        return tmp_point_sptial_distance

    def get_distance_matrix(self):
        # Create a multiprocessing pool
        st = time.time()
        print("Calculating distance matrix......")

        # Get shortest paths from center point
        st1 = time.time()
        src_node = self.point.x * self.DEM.shape[1] + self.point.y
        self.get_shortest_paths(src_node)
        ed1 = time.time()
        print("Generating shortest_paths in {:.2f} seconds".format((ed1 - st1)))
        del st1, ed1
        """
        Deprecated:Using MultiThread processing
            with Pool(processes=config.params.CORE_NUM) as p:
                distance_list = p.map(self.map_distance_matrix_new, self.dem_list)
        """
        # Using list analysis
        distance_list = [self.map_distance_matrix_new(point) for point in self.dem_list]

        ed = time.time()
        print("Done distance matrix!, using {:.2f} seconds".format((ed - st)))
        self.distance_matrix = np.array(distance_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

        return self.distance_matrix

    def map_relative_velocity_matrix(self, slope, tpoint, slope_coef):
        '''
            Calculate each point in DEM
        '''
        # print("Mapping every point in dem, getting one point relative velocity....")
        # tmp = np.cos((slope[tpoint[0], tpoint[1]] + 0.001))
        base_speed = 3200
        # if slope[tpoint[0], tpoint[1]] > 0:
        #     velocity = base_speed * np.abs(tmp)
        # elif slope[tpoint[0], tpoint[1]] < 0:
        #     velocity = base_speed / np.abs(tmp)
        # else:
        #     velocity = base_speed

        velocity = base_speed * (1 - 0.1 * np.tan(slope[tpoint[0], tpoint[1]]))

        return velocity

    def get_relative_velocity_matrix(self):
        '''
            Getting relative velocity matrix
        '''
        st = time.time()
        print("Calculating relative velocity...")

        self.slope = self.get_slope()

        # 计算坡度系数
        slope_degree = np.degrees(self.slope)

        # 设置基础速度
        base_speed = 3200
        """
        Deprecated:Using MultiThread processing
            with Pool() as p:
                funcs = partial(self.map_relative_velocity_matrix, self.slope)
                velocity_list = p.map(func=funcs, iterable=self.dem_list)
        """
        # Using List Analysis
        # velocity_list = [self.map_relative_velocity_matrix(self.slope, point,slope_coef) for point in self.dem_list]
        velocity_matrix = base_speed * np.tan(abs(self.slope))
        ed = time.time()
        print("Done relative velocity matrix!,using{:.2f} seconds".format((ed - st)))

        # self.relative_velocity_matrix = np.array(velocity_list).reshape(self.DEM.shape[0], self.DEM.shape[1])
        self.relative_velocity_matrix = velocity_matrix

    def map_final_velocity_matrix(self, tpoint):
        # Caculate the final speed by 加权平均 relative speed in paths

        # Get the target point path index of big_path
        # print("Getting target path to centerpoint...")
        dst_tpoint = tpoint[0] * self.DEM.shape[1] + tpoint[1]
        try:
            paths = self.shortest_path[dst_tpoint]
        except:
            paths = []
        relative_velocity_list = []

        # print("Mapping every point in dem, getting one point final velocity......")

        def each_point(point):
            point_x = point // self.DEM.shape[1]
            point_y = point - self.DEM.shape[1] * point_x

            return self.relative_velocity_matrix[point_x, point_y]

        if len(paths) > 0:
            relative_velocity_list.append([each_point(point) for point in paths])
            sum = np.sum(relative_velocity_list)
            if sum != 0:
                weights = np.divide(relative_velocity_list, sum)
                final_velocity = np.average(relative_velocity_list, weights=weights)
            else:
                final_velocity = 0
        else:
            final_velocity = 0

        return final_velocity

    def get_final_velocity_matrix(self):
        '''
            Calculate final velocity
        '''

        # Get relative velocity matrix first
        self.get_relative_velocity_matrix()

        st = time.time()

        """
        Deprecated:Using MultiThread processing
            with Pool() as p:
                final_velocity_list = p.map(func=self.map_final_velocity_matrix, iterable=self.dem_list)
        """
        # Using List Analysis
        final_velocity_list = [self.map_final_velocity_matrix(point) for point in self.dem_list]

        ed = time.time()
        print("Done final velocity matrix!, using {} seconds".format((ed - st)))
        self.final_velocity_matrix = np.array(final_velocity_list).reshape(self.DEM.shape[0], self.DEM.shape[1])

        return self.final_velocity_matrix

    # @DeprecationWarning
    def map_get_time_matrix(self, tpoint):
        """
        Deprecated:
        time=paths/velocity
        :param distance: 距离
        :param velocity: 速度
        :param mode_D: 距离计算模式，分为街区距离和欧式距离
        :param mode_V: 速度计算模式，暂未定
        :return: time: 时间
        """

        # Get the path between center point and each point on dem
        # paths = self.get_paths(self.point, Point(tpoint[0], tpoint[1]))
        dst_tpoint = tpoint[0] * self.DEM.shape[1] + tpoint[1]
        paths = self.shortest_path[dst_tpoint]

        total_time = 0
        # During the path calculate distance and velocity between each two point
        for index in range(len(paths) - 1):
            # tmp_point_sptial_distance = self.get_sptial_distance(Point(paths[index][0], paths[index][1]),
            #                                                      Point(paths[index + 1][0],
            #                                                            paths[index + 1][1])) * 1.2  # 修正系数
            former = paths[index]
            latter = paths[index + 1]
            tmp_point_sptial_distance = self.DEM_Graph.G[former][latter][0]['weight']

            former_x = former // self.DEM.shape[1]
            former_y = former - self.DEM.shape[1] * former_x

            latter_x = latter // self.DEM.shape[1]
            latter_y = latter - self.DEM.shape[1] * latter_x

            tmp_point_velocity = (self.relative_velocity_matrix[former_x, former_y] +
                                  self.relative_velocity_matrix[latter_x, latter_y]) / 2
            tmp_point_costtime = tmp_point_sptial_distance / tmp_point_velocity

            total_time += tmp_point_costtime

        return total_time

    # @DeprecationWarning
    def get_final_time_matrix(self):
        """
        Deprecated: Using method of distance_matrix / velocity_matrix
        :return:
        """
        #         First calculate the slope and relative_velocity_matrix
        self.get_relative_velocity_matrix()

        src_node = self.point.x * self.DEM.shape[1] + self.point.y
        self.get_shortest_paths(src_node)
        st = time.time()

        """
        Deprecated: Using MultiProcessing
            with Pool() as p:
                final_time_list = p.map(func=self.map_get_time_matrix, iterable=self.dem_list)
        """
        final_time_list = [self.map_get_time_matrix(point) for point in self.dem_list]

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
    save_dir = './test_accessDomain/mode_0_0_new/SX3_005_QMC'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # data = Image.open(file_path)
    point = Point(160, 235)
    demaxis = 30
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
    fig1, ax1 = plt.subplots()
    ax1.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, a.distance_matrix, 25)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'distance_matrix.png'), bbox_inches="tight", pad_inches=0.0)
    # #
    fig2, ax2 = plt.subplots()
    ax2.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, a.final_velocity_matrix, 25)
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'velocity_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    fig3, ax3 = plt.subplots()
    ax3.invert_yaxis()  # y轴反向
    plt.contourf(x_plt, y_plt, time_matrix, 25)
    plt.colorbar()
    # ax3.imshow(time_matrix)
    plt.savefig(os.path.join(save_dir, 'time_matrix.png'), bbox_inches="tight", pad_inches=0.0)

    kedayu = time_matrix < (30 / 60)
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

    file_path = r'G:\projects\datas\ChengGuiData\dem\Hancheng_s_new.tif'
    work_1(file_path)
    end = time.time()
    print(end - st)
