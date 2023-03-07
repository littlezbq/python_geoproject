import itertools
from functools import partial
import shutil
import time
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import argsort
import os
from PIL import Image
from pathlib import Path
import cv2 as cv
from sklearn import preprocessing

import config.params
import config.params as param


class Tools:
    def __init__(self):
        pass

    def read_dem(self, filename):
        """
        读取计算的dem文件
        :return: 返回矩阵形式的dem文件、仿射矩阵、长和宽 
        """
        dataset = gdal.Open(filename)
        im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
        im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
        # im_data = np.array(Image.open(filename))
        del dataset  # 关闭对象，文件dataset
        return im_data, im_geotrans, im_width, im_height

    def get_viewshed_file(self, filename, save_name, x, y):
        """
        计算目标dem文件中经纬度为(x,y)的可视域并保存
        :param save_name: 保存文件名
        :param x: 经度
        :param y: 纬度
        :return:
        """
        ds = gdal.Open(filename)
        band = ds.GetRasterBand(1)
        gdal.ViewshedGenerate(srcBand=band, driverName='GTiff', targetRasterName=save_name, creationOptions=None,
                              observerX=x, observerY=y,
                              observerHeight=1.7, targetHeight=0, visibleVal=255, invisibleVal=0, outOfRangeVal=0,
                              noDataVal=0, dfCurvCoeff=0.85714, mode=0, maxDistance=50000)

    def caculate_target(self, filename):
        """
        计算某一个点的可视域大小
        :return: 可视域大小
        """""
        im_data, _, _, _ = self.read_dem(filename)

        num = 0
        num = np.sum(im_data == 255, dtype=int)
        return num

    def coordinateTransform(self, filename, x, y):
        '''
            filename：输入文件名
            x：图中某点相对坐标原点（默认左上角）的列数
            y：图中某点相对坐标原点的行数
        '''
        _, im_geotrans, _, _ = self.read_dem(filename)
        start_longtitude = im_geotrans[0]
        start_latitude = im_geotrans[3]
        index_longitude = im_geotrans[1]
        index_latitude = im_geotrans[5]

        # 转换后的经纬度坐标
        px = start_longtitude + x * index_longitude + y * im_geotrans[2]
        py = start_latitude + x * im_geotrans[4] + y * index_latitude

        return [px, py], im_geotrans

    def map_file(self, filepath):
        """
        将保存下来的每个区块的可视域文件合并
        :param filepath:每个区块可视域文件所在路径
        :return: 合并后的所有可视域列表
        """
        point = []
        map_list = os.listdir(filepath)
        for f in map_list:
            result = pd.read_csv(
                filepath + '/' + f,
                sep=' ',
                header=None
            )
            point.append(result)
        concat_result = pd.concat(point, axis=0)

        return concat_result

    def save_file(self, point, savename):
        """将数组保存进文件中"""
        # if Path(savename.split('/')[0]).exists() is False:
        #     os.mkdir(savename.split('/')[0])
        point_array = np.array(point, dtype=int)
        point_sort = point_array[argsort(point_array[:, 2])]
        point_sort = point_sort[::-1]
        np.savetxt(savename, point_sort, fmt="%d %d %ld")

        return point_sort


class CalculateVisionPoint(Tools):
    def __init__(self, x_y, filename, timelimit):
        """
        计算一个可达域内所有点集的可视域并按照可视域大小排列
        :param x_y: 可达域内点集
        :param filename: dem文件
        :param timelimit: 在多长时间的可达域内计算可视域
        """
        self.x_y = x_y
        self.filename = filename
        self.timelimit = timelimit

        im_data, im_geotrans, im_width, im_height = self.read_dem(self.filename)
        self.im_width = im_width
        self.im_height = im_height
        self.im_data = im_data
        self.im_geotrans = im_geotrans

        # 不含后缀的文件名
        self.FILENAME = os.path.basename(self.filename).split('.')[0]

        self.viewsheds_path = os.path.join(param.VIEWSHED_PATH,
                                           os.path.join(self.FILENAME, str(self.timelimit)))
        self.result_savepath = os.path.join(param.ASCENSIONLIST_SAVEPATH,
                                            os.path.join(self.FILENAME, str(self.timelimit)))
        self.ascensionlist_path = os.path.join(self.result_savepath, "ascensionpoint.txt")

    def calculate_ascensionpoint_map(self, divideSpace, demAxis):
        """
        2.0：使用map()方法重构
        """

        """
        If the viewShedsPath or the resultSavePath dosen't exists, create the directory
        """
        if Path(self.viewsheds_path).exists() is False:
            os.makedirs(self.viewsheds_path)
        if Path(self.result_savepath).exists() is False:
            os.makedirs(self.result_savepath)

        """
        If viewshed files in this area is first calculated, then there's no list of denglinPoint is created, so run
        the code below
        """
        # if Path(self.ascensionlist_path).exists() is False:
        start = time.time()

        """
            First need to simplify the self.x_y list.Not every point in the reachable area need to be calculated, points
            in particular parts could have nearly the same viewshed(eg. 2 points locate between less than 50m).Therefore,
            we assume that if one point are in the square of another point(square area is 500m*500m),then this point can
            be jumped over.

        """
        domainPointsSimplifed = []

        # Setting distance
        disreal = divideSpace
        discur = int(disreal / demAxis)
        # Should be in this way
        # discur = int(disreal / max(xais,yais))

        # Getting the min and max value for row and col
        maxX, maxY = np.max(self.x_y, axis=0)
        minX, minY = np.min(self.x_y, axis=0)

        # Setting base points
        baseXList = list(range(minX, maxX, discur))
        baseYList = list(range(minY, maxY, discur))

        baseCorrds = np.array([[i, k] for i in baseXList for k in baseYList])

        # Check if baseCorrds in domainPointLists. If in, then add to the simplifed list
        for corrd in baseCorrds:
            if (corrd == self.x_y).all(1).any():
                domainPointsSimplifed.append(corrd)

        # Calculate the point in domainPointsSimplifed list using map()
        result_list = list(map(self.cal_map, domainPointsSimplifed))

        result_sort = self.save_file(result_list, self.ascensionlist_path)

        # 将登临点的可视域文件复制到结果文件夹中,默认最多可以展示前100个

        for i in range(100):
            denglinPointViewShed = Image.open(os.path.join(self.viewsheds_path,

                                                           "".join((str(result_sort[i][0]),
                                                                    '_', str(result_sort[i][1]), '.tif'))))
            denglinPoint_viewshed_path = os.path.join(self.result_savepath,
                                                      "".join((str(result_sort[i][0]),
                                                               '_', str(result_sort[i][1]),
                                                               'ascensionpoint_viewshed.png')))
            denglinPointViewShed.save(denglinPoint_viewshed_path)

        #   保留前100个可视域文件
        self.saveNo_100Files()
        # 加载本地已经计算出的登临点列表
        # else:
        #     result_sort = np.loadtxt(os.path.join(self.result_savepath, "ascensionpoint.txt"))

        end = time.time()

        print("Using {:.2f}".format(end - start))

        return np.array(result_sort[0:100])


    def cal_map(self, point):
        """
        Calculate one corrd viewshed(base on map method)
        :param point:
        :return:
        """
        saveName = os.path.join(self.viewsheds_path, "".join((str(point[0]), '_', str(point[1]), '.tif')))

        start_longtitude = self.im_geotrans[0]
        start_latitude = self.im_geotrans[3]
        index_longitude = self.im_geotrans[1]
        index_latitude = self.im_geotrans[5]

        x = point[0]
        y = point[1]

        px = start_longtitude + x * index_longitude + y * self.im_geotrans[2]
        py = start_latitude + x * self.im_geotrans[4] + y * index_latitude

        self.get_viewshed_file(self.filename, saveName, px, py)

        # Calculate the num of viewshed
        totalViewshedPoint = self.caculate_target(saveName)

        # Save the coord and num of viewshed into list
        return [x, y, totalViewshedPoint]

    def calculate_ascensionpoint_mp(self, divideSpace, demAxis):
        """
        Using
        Multiprocess method after optimization: First need to simplify the self.x_y list.Not every point in the
        reachable area need to be calculated, points in particular parts could have nearly the same viewshed(eg. 2
        points locate between less than 50m).Therefore, we assume that if one point are in the square of another
        point(square area is 500m*500m),then this point can be jumped over.
        """
        if os.path.exists(self.viewsheds_path) is False:
            os.makedirs(self.viewsheds_path)
        if os.path.exists(self.result_savepath) is False:
            os.makedirs(self.result_savepath)

        start = time.time()


        # Check if viewshed files already generated
        for file in os.listdir(self.viewsheds_path):
            if file.endswith(".tif") is True and os.path.exists(self.ascensionlist_path) is True:
                result = np.loadtxt(self.ascensionlist_path)
                return np.array(result[0:100])

        domainPointsSimplifed = []

        # 设置间隔距离，单位为米
        disreal = divideSpace
        discur = int(disreal / demAxis)

        # Getting the min and max value for row and col
        maxX, maxY = np.max(self.x_y, axis=0)
        minX, minY = np.min(self.x_y, axis=0)

        # Setting base points
        baseXList = list(range(minX, maxX, discur))
        baseYList = list(range(minY, maxY, discur))

        baseCorrds = np.array([[i, k] for i in baseXList for k in baseYList])

        # Check if baseCorrds in domainPointLists. If in, then add to the simplifed list
        for corrd in baseCorrds:
            if (corrd == self.x_y).all(1).any():
                domainPointsSimplifed.append(corrd)

        with mp.Pool(processes=config.params.CORE_NUM) as pool:
            x_y_compute = domainPointsSimplifed

            func = partial(self.cal_mp, self.filename, self.viewsheds_path,
                           self.im_geotrans)

            print("Calculate Begin...")

            point = pool.map(func, x_y_compute)

            print()

            end = time.time()
            print("Using {:.2f} seconds".format((end - start)))

        result_sort = self.save_file(point, self.ascensionlist_path)

        # 将登临点的可视域文件复制到结果文件夹中,默认最多可以展示前100个
        for i in range(100):
            denglinPointViewShed = Image.open(os.path.join(self.viewsheds_path,
                                                           "".join((str(result_sort[i][0]),
                                                                    '_', str(result_sort[i][1]), '.tif')))).convert(
                'RGB')
            denglinPoint_viewshed_path = os.path.join(self.result_savepath,
                                                      "".join((str(result_sort[i][0]),
                                                               '_', str(result_sort[i][1]),
                                                               'ascensionpoint_viewshed.png')))
            denglinPointViewShed.save(denglinPoint_viewshed_path)

        # 加载本地已经计算出的登临点列表
        # else:
        #     result_sort = np.loadtxt(os.path.join(self.result_savepath, "ascensionpoint.txt"))

        return np.array(result_sort[0:100])

    def cal_mp(self, filename, middlepath, im_geotrans, pos):
        """
            计算一个点的可视域
            filename：输入文件名
        """
        start_longtitude = im_geotrans[0]
        start_latitude = im_geotrans[3]
        index_longitude = im_geotrans[1]
        index_latitude = im_geotrans[5]

        # 传入的pos第一维是行数，第二维是列数，因此需要变换一下坐标，满足坐标转换时的经纬度顺序
        x = pos[1]#列
        y = pos[0]#行

        px = start_longtitude + x * index_longitude + y * im_geotrans[2]
        py = start_latitude + x * im_geotrans[4] + y * index_latitude
        # 设置存储文件名字
        savename = "".join((middlepath + '/' + str(x) + '_' + str(y) + '.tif'))
        # 计算并存储可视域文件
        # print("Generating " + savename)
        self.get_viewshed_file(filename, savename, px, py)
        # 计算出可视域文件中可视点的多少
        total_viewshed_point = self.caculate_target(savename)

        # 将该点的相对坐标和可视点大小存入列表    
        return [x, y, total_viewshed_point]

    def calculate_exposivepoint(self, timelimit):
        """
            timelimit: 步行时长，单位为分钟
            exposiveNum: 需要显示的暴露点数量

            return: 返回原图大小的可视域叠加图
        """

        # 读取时间矩阵，重新导入可达域
        timeMatrixPath = os.path.join(param.REACHABLEIMG_PATH, self.FILENAME)
        timeMatrix = np.loadtxt(timeMatrixPath + "/time_matrix.txt")
        ts_3600 = timeMatrix < (timelimit / 60.0)

        # Initialize result matrix
        result = np.zeros([self.im_height, self.im_width])

        try:
            for viewshedFile in os.listdir(self.viewsheds_path):
                # 读取可视域文件
                if viewshedFile.endswith(".tif"):
                    currentfile = os.path.join(self.viewsheds_path, viewshedFile)
                    viewshed = np.array(Image.open(currentfile))
                    #   将可达域内各视点的可视域叠加
                    result += viewshed
        except:
            print("No viewshed file detected")
            exit(-1)

        # 只计算可达域内点,将result数组中可达域外点位的可视域置为0
        result[ts_3600 == False] = 0

        # 归一化result数组中可达域内点位的叠加可视域至0-255
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255), copy=True)
        result_new = min_max_scaler.fit_transform(result)

        return result,result_new

    def saveNo_100Files(self):
        """
            保存前100个登临点的可视域文件
        """

        #       删除登临点列表中100名之后的可视域文件
        ascensionlist = np.loadtxt(self.ascensionlist_path)

        for i in range(100, len(ascensionlist), 1):
            x = ascensionlist[i][0]
            y = ascensionlist[i][1]
            # 删除该文件
            os.remove(os.path.join(self.viewsheds_path, "".join((str(int(x)), '_', str(int(y)), '.tif'))))

        print("Deleted extra viewshed files!")
