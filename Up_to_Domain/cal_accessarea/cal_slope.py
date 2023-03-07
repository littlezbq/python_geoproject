import numpy as np
from osgeo import gdal
import cv2 as cv

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
        slope = np.arctan(np.sqrt(slope_we * slope_we + slope_sn * slope_sn))
        return slope

    @DeprecationWarning
    def CacSlopAsp(self, dx, dy):
        """
        Deprecated:
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
