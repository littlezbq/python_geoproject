import os

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from config.params import *

from config.params import *


# 保存村落平面形态解析的遥感图和掩膜图
def save_raw_quantization(raw_remote, raw_mask):

    # 目标结果的存放路径
    remote_path = os.path.join(VILLAGE_SPACE_QUANTIZATION,'drawremote.png')
    mask_path = os.path.join(VILLAGE_SPACE_QUANTIZATION,'drawrawmask.png')

    # 用OpenCV读取原始遥感数据和掩膜图为两个对象（矩阵）
    remote = cv.imread(raw_remote)
    mask = cv.imread(raw_mask)
    # 将两个对象保存到新的路径下
    cv.imwrite(remote_path,remote)
    cv.imwrite(mask_path,mask)

    # remote_path = "/api/" + remote_path
    # mask_path = "/api/" + mask_path

    return remote_path, mask_path




