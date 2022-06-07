from osgeo import gdal
import numpy as np
from PIL import Image
from calculate_denglinPoint.utils.cdld import Tools

dem_path = "static/datasets/SX3_005_QMC.tif"
tl = Tools()
im_data, im_geotrans, im_width, im_height = tl.read_dem(filename=dem_path)

[px,py] = tl.coordinateTransform(dem_path,255,0)

print(px,py)


print(im_geotrans)
pass