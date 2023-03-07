# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
import shutil
import argparse


def cvt_one(json_path, img_path, save_path, label_color):
    # load img and json
    data = json.load(open(json_path, encoding='gbk'))
    img = cv2.imread(img_path)

    # get background data
    img_h = data['imageHeight']
    img_w = data['imageWidth']
    color_bg = (0, 0, 0)
    points_bg = [(0, 0), (0, img_h), (img_w, img_h), (img_w, 0)]
    img = cv2.fillPoly(img, [np.array(points_bg)], color_bg)

    # draw roi
    for i in range(len(data['shapes'])):
        name = data['shapes'][i]['label']
        points = data['shapes'][i]['points']
        color = data['shapes'][i]['fill_color']
        # data['shapes'][i]['fill_color'] = label_color[name]  # 修改json文件中的填充颜色为我们设定的颜色
        # color = data['shapes'][i]['fill_color']
        if label_color:
            img = cv2.fillPoly(img, [np.array(points, dtype=int)], label_color[name])
        else:
            img = cv2.fillPoly(img, [np.array(points, dtype=int)], (color[0], color[1], color[2]))
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    save_dir = 'datasets'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_dir = 'datasets'
    files = os.listdir(file_dir)
    img_files = list(filter(lambda x: '.png' in x, files))
    label_color = {  # 设定label染色情况
        # 'camel': (225, 225, 255),
        # 'pyramid': (225, 225, 255),
        # 'stick': (225, 225, 255),
        # 'danxiashan': (225, 225, 255),
        # 'xiangbishan': (225, 225, 255),
        # 'wuzhishan': (225, 225, 255),
        # 'mantoushan': (225, 225, 255),
        # 'mushroom':(255,255,255),
        # 'volcano':(255,255,255)
        # 'guifengshan':(255,255,255)
        'river':(255,255,255)
    }
    save_img = 'img'
    if not os.path.exists(save_img):
        os.makedirs(save_img)
    for i in range(len(img_files)):
        img_path = file_dir + '/' + img_files[i]
        shutil.copy(img_path, save_img + '/' + img_files[i])

        json_path = img_path.replace('.png', '.json')
        print(json_path)
        save_path = save_dir + '/' + img_files[i]
        print('Processing {}'.format(img_path))
        cvt_one(json_path, img_path, save_path, label_color)

    # 重命名jpg为png
    # file_dir = 'origin'
    # files = os.listdir(file_dir)
    # for file in files:
    #     if file.endswith('.jpg'):
    #         shutil.copy(os.path.join(file_dir,file),os.path.join(file_dir,os.path.basename(file).split('.')[0]+'.png'))
