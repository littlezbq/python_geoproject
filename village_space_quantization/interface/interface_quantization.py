from cmath import sqrt
import os
import cv2
import numpy as np
import math
from config.params import *


def interface_analysis_shape(mask_file_name):
    # 边界形状

    # 目标结果的存放路径
    result_path = os.path.join(VILLAGE_SPACE_QUANTIZATION, 'result_shape.png')
    raw_mask = os.path.join(VILLAGE_SPACE_QUANTIZATION,mask_file_name+'.png')
    # 用OpenCV读取原始掩膜图
    image = cv2.imread(raw_mask)
    # 将原始的掩膜图存到新的路径下
    cv2.imwrite(os.path.join(VILLAGE_SPACE_QUANTIZATION, 'raw_shapemask.png'), image)

    # 具体的处理过程
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界

        # 找面积最小的矩形
        rect = cv2.minAreaRect(c)
        # 得到最小矩形的坐标
        box = cv2.boxPoints(rect)
        # 标准化坐标到整数
        box = np.int0(box)
        # 画出边界
        cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
        # 计算长宽比
        aspect_ratio = float(w) / h
        # 计算轮廓面积
        A = abs(cv2.contourArea(c, True))
        # 计算轮廓周长
        P = cv2.arcLength(c, True)
        # 计算形状指数
        S = (P / 1.5 * aspect_ratio - sqrt(aspect_ratio) + 1.5) * sqrt(aspect_ratio / A * math.pi)

    # 处理结果
    shape_result = cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    # 处理结果存在与原始掩膜图相同的路径下
    cv2.imwrite(result_path, shape_result)
    # 返回一个新的路径到run.py，需上传至前端
    # result_path = "/api/" + result_path
    return result_path


def interface_analysis_space(mask_file_name):
    # 空间结构

    # 目标结果的存放路径
    result_path = os.path.join(VILLAGE_SPACE_QUANTIZATION, 'result_space.png')
    raw_mask = os.path.join(VILLAGE_SPACE_QUANTIZATION, mask_file_name + '.png')
    # 用OpenCV读取原始掩膜图
    image = cv2.imread(raw_mask)
    # 将原始的掩膜图存到新的路径下
    cv2.imwrite(os.path.join(VILLAGE_SPACE_QUANTIZATION, 'raw_spacemask.png'), image)

    # 具体的处理过程
    # 提取图中的红色部分
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img2=cv2.drawContours(mask, contours, -1, (0,255,0), 3)

    for c in contours:
        # cnt=contours[0]
        # 计算轮廓面积
        A = abs(cv2.contourArea(c, True))
        # 计算轮廓周长
        P = cv2.arcLength(c, True)
        # 计算分形数值
        D1 = math.log(P / 4) / math.log(10)
        D2 = math.log(A) / math.log(10)
        D = 2 * D1 / D2

    # cv2.imshow("test",mask)
    # cv2.imwrite('sava2.png', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 处理结果
    space_result = 255 - mask
    # 处理结果存在与原始掩膜图相同的路径下
    cv2.imwrite(result_path, space_result)
    # 返回一个新的路径到run.py，需上传至前端

    # result_path = "/api/" + result_path
    return result_path


def interface_analysis_order(mask_file_name):
    # 建筑顺序

    # 目标结果的存放路径
    result_path = os.path.join(VILLAGE_SPACE_QUANTIZATION, 'result_order.png')
    raw_mask = os.path.join(VILLAGE_SPACE_QUANTIZATION, mask_file_name + '.png')
    # 用OpenCV读取原始掩膜图
    image = cv2.imread(raw_mask)
    # 将原始的掩膜图存到新的路径下
    cv2.imwrite(os.path.join(VILLAGE_SPACE_QUANTIZATION, 'raw_ordermask.png'), image)

    # 具体的处理过程
    # 提取红色区域建筑物轮廓
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # 获取中心点
        M = cv2.moments(contours[i])
        cx = int(M['m10'] / M['m00'])  # 求x坐标
        cy = int(M['m01'] / M['m00'])  # 求y坐标

    # 处理结果
    order_result = cv2.circle(image, (cx, cy), 3, 128, -1)
    # 处理结果存在与原始掩膜图相同的路径下
    cv2.imwrite(result_path, order_result)
    # 返回一个新的路径到run.py，需上传至前端
    # result_path = "/api/" + result_path
    return result_path
