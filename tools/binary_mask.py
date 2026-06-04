import cv2
import numpy as np
import shutil


def binary_mask_128_with_backup(image_path):
    """
    二值化并备份原图
    """

    # 处理
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = np.where(img > (255*0.75), 255, 0).astype(np.uint8)
    cv2.imwrite(image_path, binary)

    # print(f"二值化完成：{image_path}")
    return binary

import os
n = 0
path = "/home/yassine/projects/UACANet-main/results_cl/Polyp-PVT-results/PolypPVT-0.75"
# for imgname in os.listdir(path):
#     imgpath = os.path.join(path, imgname)
#     n+=1
#     binary_mask_128_with_backup(imgpath)
for dataname in os.listdir(path):
    datapath = os.path.join(path, dataname)
    for imgname in os.listdir(datapath):
        imgpath = os.path.join(datapath, imgname)
        n+=1
        binary_mask_128_with_backup(imgpath)
print(n)