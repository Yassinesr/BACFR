import os
import numpy as np
from PIL import Image


def simple_iou_comparison(folder1, folder2):
    """简化的IoU计算版本"""

    # 获取两个文件夹中的PNG文件
    files1 = [f for f in os.listdir(folder1) if f.endswith('.png')]



    iou_scores = []

    for filename in files1:
        # 加载掩码


        mask1 = np.array(Image.open(os.path.join(folder1, filename)))
        mask2 = np.array(Image.open(os.path.join(folder2, filename)))

        # 转换为二值
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)

        # 计算IoU
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)


    if iou_scores:
        print(f"\n平均IoU: {np.mean(iou_scores):.4f}")

    return iou_scores


# 使用示例
if __name__ == "__main__":
    # folder1 = "results/UACAPatchNet-one_stage-2-2"
    # folder2 = "/home/yassine/projects/BPR/dataset/patches"
    # f1 = folder2 + '/' + 'mask_dir'+'/'+'train'
    # f2 = folder2 + '/' + 'ann_dir'+'/'+'train'

    f1 = '../dataset/pranet-traindataset/PatchesDataset-0.9603_644/mask_dir/train'
    f2 = '../dataset/pranet-traindataset/PatchesDataset-0.9603_644/ann_dir/train'
    iou_scores = simple_iou_comparison(f1, f2)