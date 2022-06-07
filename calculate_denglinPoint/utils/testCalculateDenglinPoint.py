import time
from tkinter import Image
from PIL import Image
import numpy as np
from pathlib import Path
import os
from osgeo import gdal


def read_dem(filename):
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

def get_viewshed_file(filename, im_geotrans, save_name, x, y):
    """
    计算目标dem文件中经纬度为(x,y)的可视域并保存
    :param save_name: 保存文件名
    :param x: 经度
    :param y: 纬度
    :return:
    """
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)

    start_longtitude = im_geotrans[0]
    start_latitude = im_geotrans[3]
    index_longitude = im_geotrans[1]
    index_latitude = im_geotrans[5]

    px = start_longtitude + x * index_longitude + y * im_geotrans[2]
    py = start_latitude + x * im_geotrans[4] + y * index_latitude


    gdal.ViewshedGenerate(srcBand=band, driverName='GTiff', targetRasterName=save_name, creationOptions=None,
                            observerX=px, observerY=py,
                            observerHeight=2, targetHeight=0, visibleVal=255, invisibleVal=0, outOfRangeVal=0,
                            noDataVal=0, dfCurvCoeff=0.85714, mode=2, maxDistance=0)


def cal(p):
    savaName = os.path.join('resultUsingMapFunction',
                                    "".join((str(p[0]),'_',str(p[1]),".tif")))

    testImgPath = "test.tif"
    testImg, testImgGeoTrans, _, _ = read_dem(testImgPath)


    get_viewshed_file(testImgPath,testImgGeoTrans, savaName, p[0], p[1])


if __name__ == "__main__":

    testImgPath = "test.tif"
    dataset = gdal.Open(testImgPath)
    im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
    im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)  # 将数据写成数组，对应栅格矩阵
    # im_data = np.array(Image.open(filename))
    del dataset  # 关闭对象，文件dataset

    testDir = "resultUsingMapFunction"

    
    x_y = np.loadtxt("domainPointIndex.txt", dtype=int)
    x_y_read = np.array([(x_y[:, 1][i], x_y[:, 0][i]) for i in range(len(x_y))])

    start = time.time()
    result = list(map(cal,list(x_y_read)))
    end = time.time()
    print(end - start)
