import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image


def draw_ascensionpoint(dem_path, remote_path, ascensionpointindex_path):
    ascension_index = np.loadtxt(ascensionpointindex_path)
    remote = Image.open(remote_path)
    dem = Image.open(dem_path)

    X_rules, Y_rules = remote.size[0] / dem.size[0], remote.size[1] / dem.size[1]

    three_points = [ascension_index[i] for i in range(3)]

    remote_data = np.asarray(remote)
    for point in three_points:
        cv.circle(remote_data, (int(point[0] * Y_rules), int(point[1] * X_rules)), 50, (255, 0, 0))
    # cv.circle(remote_data, (1000,5),50,(255,0,0))
    cv.imwrite("drawPic.png", remote_data)


def quality_ascensionpoint(dem_path, ascensionpointindex_path):
    dem_data = np.asarray(Image.open(dem_path))
    ascension_index = np.loadtxt(ascensionpointindex_path)
    total_points = dem_data.shape[0] * dem_data.shape[1]

    three_points = [ascension_index[i] for i in range(3)]

    for point in three_points:
        ascensionpoint_area = point[2]
        print("Vision area percent: {}.2f".format(ascensionpoint_area * 100 / total_points))


if __name__ == "__main__":
    # draw_ascensionpoint(dem_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.tif",
    #                     remote_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.png",
    #                     ascensionpointindex_path=r"D:\projects\webGisProject_backEnd\static\result\denglin\SX3_005_QMC\15\ascensionpoint.txt"
    #                     )

    quality_ascensionpoint(dem_path=r"D:\projects\webGisProject_backEnd\static\datasets\SX3_005_QMC.tif",
                           ascensionpointindex_path=r"D:\projects\webGisProject_backEnd\static\result\denglin\SX3_005_QMC\15\ascensionpoint.txt"
                           )
