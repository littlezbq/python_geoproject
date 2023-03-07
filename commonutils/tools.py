import os
import shutil
import numpy as np
import pandas as pd
from osgeo import gdal
from numpy.core.fromnumeric import argsort


def removing(current_path):
    for file in os.listdir(current_path):
        current_path_file = os.path.join(current_path, file)
        if os.path.isdir(current_path_file):
            shutil.rmtree(current_path_file)
        elif os.path.isfile(current_path_file):
            os.remove(current_path_file)


class Tools:
    def __init__(self):
        pass

    # Clear caches in calculating ascension point
    @staticmethod
    def clear_cache(filename="*", time_limit="*"):
        import config.params as param1

        if filename == "*":
            current_path = param1.VIEWSHED_PATH
            removing(current_path)

        elif filename != "*" and time_limit == "*":

            current_path = os.path.join(param1.VIEWSHED_PATH,
                                        str(filename))
            # 删除filename下所有步行时间的结果
            removing(current_path)

        elif filename != "*" and time_limit != "*":

            current_path = os.path.join(param1.VIEWSHED_PATH,
                                        os.path.join(str(filename),
                                                     str(time_limit)))
            # 删除某个步行时间的所有结果
            removing(current_path)

    # 清除结果文件，需要指定清除可达域计算结果还是登临点计算结果
    @staticmethod
    def clear_result(mode=0, filename="*", time_limit="*"):
        import config.params as param

        def clear_denglin(filename="*", time_limit="*"):

            # 删除./static/result/denglin下所有文件
            if filename == "*":
                current_path = param.ASCENSIONLIST_SAVEPATH
                removing(current_path)

            # 删除./static/result/denglin/filename下所有文件

            elif filename != "*" and time_limit == "*":
                current_path = os.path.join(param.ASCENSIONLIST_SAVEPATH, str(filename))
                removing(current_path)

            # 删除./static/result/denglin/filename/time_limit下所有文件

            elif filename != "*" and time_limit != "*":

                current_path = os.path.join(param.ASCENSIONLIST_SAVEPATH,
                                            os.path.join(str(filename),
                                                         str(time_limit)))
                removing(current_path)

        def clear_reachableArea(filename="*", time_limit="*"):
            # 暂时只清除可达域计算结果，保留源数据的展示结果
            # 清除reachableImg目录下所有村落的子文件夹
            if filename == "*":
                current_path = param.REACHABLEIMG_PATH
                removing(current_path)

            # 清除reachableImg/filename下所有文件
            elif filename != "*" and time_limit == "*":
                current_path = os.path.join(param.REACHABLEIMG_PATH, str(filename))
                removing(current_path)

        #     默认清除可达域和登临点所有计算结果
        if mode == 0:
            #         清除登临点文件
            clear_denglin()
            #         清除可达域文件
            clear_reachableArea()

        # 当mode不为0时，清除登临点和可达域目录下的filename和time_limit是不同的，不会冲突
        elif mode == 1:
            # 仅清除登临点
            clear_denglin(filename, time_limit)
        elif mode == 2:
            # 仅清除可达域
            clear_reachableArea(filename, time_limit)

    @staticmethod
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

    @staticmethod
    def get_viewshed_file(filename, save_name, x, y):
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
                              observerHeight=2, targetHeight=0, visibleVal=255, invisibleVal=0, outOfRangeVal=0,
                              noDataVal=0, dfCurvCoeff=0.85714, mode=0, maxDistance=50000)

    @staticmethod
    def caculate_target(filename):
        """
        计算某一个点的可视域大小
        :return: 可视域大小
        """""
        im_data, _, _, _ = Tools.read_dem(filename)

        num = np.sum(im_data == 255, dtype=int)
        return num

    @staticmethod
    def coordinateTransformOfPoint(filename, x, y):
        '''
            filename：输入文件名
            x：图中某点相对坐标原点（默认左上角）的列数
            y：图中某点相对坐标原点的行数
        '''
        _, im_geotrans, _, _ = Tools.read_dem(filename)
        start_longtitude = im_geotrans[0]
        start_latitude = im_geotrans[3]
        index_longitude = im_geotrans[1]
        index_latitude = im_geotrans[5]

        # 转换后的经纬度坐标
        px = start_longtitude + x * index_longitude + y * im_geotrans[2]
        py = start_latitude + x * im_geotrans[4] + y * index_latitude

        return [px, py], im_geotrans

    @staticmethod
    def map_file(filepath):
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

    @staticmethod
    def save_file(point, savename):
        """将数组保存进文件中"""
        # if Path(savename.split('/')[0]).exists() is False:
        #     os.mkdir(savename.split('/')[0])
        point_array = np.array(point, dtype=int)
        point_sort = point_array[argsort(point_array[:, 2])]
        point_sort = point_sort[::-1]
        np.savetxt(savename, point_sort, fmt="%d %d %ld")

        return point_sort

    @staticmethod
    def coordinateTransformOfFile(srcFile,dstFile):
        """
        Transform file from UTM coordinate to GEO coordinate, using resample nearest
        :param srcFile:
        :param dstFile:
        :return:
        """
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        # 打开源数据文件
        with rasterio.open(srcFile) as src:
            # 获取源数据的地理变换信息、投影坐标系和像素大小
            src_transform = src.transform
            src_crs = src.crs
            src_res = src.res

            # 使用WKT字符串来定义目标数据的空间参考系
            dst_crs = rasterio.crs.CRS.from_wkt(
                'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')

            # 计算目标数据的地理变换信息和像素大小，分辨率设置为源数据的分辨率
            dst_transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height,
                                                                       dst_width=src.width, dst_height=src.height,
                                                                       *src.bounds)

            # 创建目标数据文件
            dst_profile = src.profile
            dst_profile.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': width,
                'height': height,
                'dtype': 'int16',  # 设置数据类型为int16
                'nodata': -9999,  # 设置NoData值为-9999
            })

            with rasterio.open('dstFile', 'w', **dst_profile) as dst:
                # 进行投影变换
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
