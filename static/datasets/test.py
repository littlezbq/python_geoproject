from osgeo import gdal
from PIL import Image
from pathlib import Path
import cv2 as cv
import numpy as np


def show_metadata(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 读取图像的宽度，x方向上的像素个数
    im_height = dataset.RasterYSize  # 读取图像的高度，y方向上的像素个数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵

    west = im_geotrans[0]
    south = im_geotrans[3] + im_height*im_geotrans[-1]
    east = im_geotrans[0] + im_width*im_geotrans[1]
    north = im_geotrans[3]


    print(west, south, east, north)

def draw_dem(filename):
    demdata = np.array(Image.open(filename).convert('L'))
    filenameobj = Path(filename)
    newname = str(filenameobj.parent) + "/" + filenameobj.stem + "new.png"
    
    cv.imwrite(newname,demdata)



if __name__ == "__main__":
    filename = "static/datasets/Yanan.tif"
    show_metadata(filename)



