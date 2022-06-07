from .unet import Unet
# from unet import Unet
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2 as cv


def interface_villageEdgeDect(remote):
    """
        传入待分析村落遥感数据
        返回村落边界掩膜、叠加图和村落几何中心
    """

    dir_save_path = "./static/result/village_edge"
    # dir_save_path = "./test_village_edge"
    if os.path.exists(dir_save_path) is False:
        os.makedirs(dir_save_path)

    villageEdge_name = os.path.basename(remote).split('.')[0] + 'Edge.png'
    conbine_name = os.path.basename(remote).split('.')[0] + 'Combine.png'

    save_villageEdge_name_dir = dir_save_path + '/' + villageEdge_name
    save_conbineName_dir = dir_save_path + '/' + conbine_name
    edge_center = dir_save_path + '/' + os.path.basename(remote).split('.')[0] + 'Combine1.png'

    if os.path.exists(save_villageEdge_name_dir) is False:
        # 获得村落边界,数据为pillow格式
        unet = Unet()
        remoteImg = Image.open(remote)
        villageEdge, oldImg = unet.detect_image(remoteImg)

    else:
        villageEdge = Image.open(save_villageEdge_name_dir)
        oldImg = Image.open(remote)

    combineImage = np.array(Image.blend(oldImg, villageEdge, 0.3))
    # combineImage = cv.addWeighted(oldImg, 0.7, villageEdge, 0.3, 0)

    # 获取村落中心
    x_mean, y_mean = calculate_village_center(villageEdge)

    # 分别保存掩膜图和叠加图,以及包含村落中心的叠加图
    cv.imwrite(save_conbineName_dir, combineImage)
    # cv.imwrite(save_villageEdge_name_dir, villageEdge)
    villageEdge.save(save_villageEdge_name_dir)

    cv.circle(combineImage, (x_mean.astype(int), y_mean.astype(int)), 80, (0, 255, 0), 4)
    cv.imwrite(edge_center, combineImage)

    # return save_villageEdge_name_dir, save_conbineName_dir, edge_center, [x_mean, y_mean]
    return [x_mean, y_mean]


def calculate_village_center(village_edge):
    """
        传入村落边界掩膜图
        返回依照遥感图像计算出的村落中心与掩模图的叠加
    """

    img = np.array(village_edge)
    #   先找出村落边界内部点的坐标
    index = np.argwhere(img == 128)

    #   分别求x和y方向的均值
    x_mean = np.ceil(np.mean(index[:, 1])).astype(int)
    y_mean = np.ceil(np.mean(index[:, 0])).astype(int)

    return x_mean, y_mean


if __name__ == "__main__":
    interface_villageEdgeDect(
        r"D:\projects\webGisProject-dev20220412\static\datasets\SX3_005_QMC.png")
    # calculate_village_center(r"F:\Projects\Pytorch\pytorch-flask-api-vue-elementUi_2021-11-4-1837\webGisProject\village_edge\img_out\test.png")
