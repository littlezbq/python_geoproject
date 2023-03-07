import cv2 as cv
import numpy as np

def threshold_with_otsu(img):

    # 将原图像范围扩展到0-255
    max = img[np.unravel_index(np.argmax(img, axis=None), img.shape)]
    if max < 255:
        img = img * 255
        img = img.astype("uint8")

    ret, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return ret, th


def top_hat(img):
    kernel = np.ones((3, 3), np.uint8)
    tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

    return tophat



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "../datas/test.png"
    img = cv.imread(path)
    img = cv.cvtColor(img, code=cv.COLOR_RGB2GRAY)

    # _, res = threshold_with_otsu(img)
    res = top_hat(img)



    plt.figure()
    plt.imshow(res)
    plt.show()
