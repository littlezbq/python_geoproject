from road_intersection.road_extraction.utils import cvtColor, resize_image, preprocess_input
import numpy as np
import copy
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
# from road_intersection.road_extraction.nets import unet
from road_intersection.road_extraction.nets.dense_unet import DenseUNet
# 需要修改的路径参数
# model_path = "road_intersection/road_extraction/model_data/best_epoch_weights.pth"
model_path = r"G:\projects\webGisProject_backEnd20220903\road_intersection\road_extraction\model_data\best_epoch_weights.pth"

#   所需要区分的类的个数+1
num_classes = 2

#   所使用的的主干网络：vgg、resnet50
# backbone = "resnet50"

#   输入图片的大小
input_shape = [1024, 1024]

#   mix_type = 0的时候代表原图与生成的图进行混合
#   mix_type = 1的时候代表仅保留生成的图
#   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
mix_type = 1

cuda = True

colors = [(0, 0, 0), (128, 0, 0)]


def get_road(image, count=False, name_classes=None):
    '''
    Get road from RGB image
    :param image: input image contains road
    :param count:
    :param name_classes:
    :return: Binary road graph
    '''
    # Turn image into RGB mode
    image = cvtColor(image)
    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        # model = unet.Unet(num_classes=num_classes, backbone=backbone) #使用unet检测道路
        model = DenseUNet(n_classes=num_classes).eval() #使用DenseUnet-sp检测道路
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.eval()
        print('{} model, and classes loaded.'.format(model_path))

        model = model.cuda()

        pr = model(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        classes_nums = np.zeros([num_classes])
        total_points_num = orininal_h * orininal_w
        print('-' * 63)
        print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        print('-' * 63)
        for i in range(num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            if num > 0:
                print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                print('-' * 63)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)

    if mix_type == 0:

        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))
        # ------------------------------------------------#
        #   将新图与原图及进行混合
        # ------------------------------------------------#
        image = Image.blend(old_img, image, 0.7)

    elif mix_type == 1:
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

    elif mix_type == 2:
        seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image = Image.open(r"D:\projects\webGisProject_backEnd20220903\road_intersection\datas\AH1_003_YF.png")

    res = get_road(image)

    plt.figure()
    plt.imshow(res)
    plt.show()
