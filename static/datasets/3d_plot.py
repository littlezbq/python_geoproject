# -*- coding: gbk -*-
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

gdal.AllRegister()

filePath = u"E:\����\demʵ��/Himalaya.tif"  # �������dem����
dataset = gdal.Open(filePath)
adfGeoTransform = dataset.GetGeoTransform()
band = dataset.GetRasterBand(1)  # ��gdalȥ��д������ݣ���Ȼdemֻ��һ������
nrows = dataset.RasterXSize
ncols = dataset.RasterYSize  # �������о��Ƕ�ȡ���ݵ�������
Xmin = adfGeoTransform[0]  # ������ݵ�ƽ������
Ymin = adfGeoTransform[3]
Xmax = adfGeoTransform[0] + nrows * adfGeoTransform[1] + ncols * adfGeoTransform[2]
Ymax = adfGeoTransform[3] + nrows * adfGeoTransform[4] + ncols * adfGeoTransform[5]
x = np.linspace(Xmin, Xmax, ncols)
y = np.linspace(Ymin, Ymax, nrows)
X, Y = np.meshgrid(x, y)
Z = band.ReadAsArray(0, 0, nrows, ncols)  # ��һ�ξ��ǽ����ݵ�x��y��z����numpy����
region = np.s_[10:400, 10:400]
X, Y, Z = X[region], Y[region], Z[region]
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12, 10))
ls = LightSource(270, 20)  # ��������ӻ����ݵ�ɫ��
rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
plt.show()  # �����Ⱦ����ÿ�����άͼ��
