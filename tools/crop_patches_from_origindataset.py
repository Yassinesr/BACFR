import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.ops import roi_align
from mmcv.ops.nms import nms


# ========================
# Helper functions
# ========================
def calculate_iou(pred_mask, gt_mask):
    """
    计算两个二值mask之间的IOU
    pred_mask, gt_mask: 二值tensor (H, W)
    """
    intersection = torch.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum().float()
    union = torch.logical_or(pred_mask > 0.5, gt_mask > 0.5).sum().float()

    if union == 0:
        return torch.tensor(0.0, device=pred_mask.device)
    return intersection / union


def _force_move_back(sdets, H, W, patch_size):
    # Make sure patch boxes stay inside the image.
    # "sdets" are boxes like [x1,y1,x2,y2] (may also include a score column).
    # If a box would go outside the image, move it so the whole patch fits.
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets


def _to_rois(boxes):
    # Prepare boxes for roi_align. roi_align expects each box to start with a
    # batch index like [batch_idx, x1, y1, x2, y2]. We only have one image,
    # so batch index is 0 for every box.
    idx = torch.zeros((boxes.size(0), 1), device=boxes.device)
    return torch.cat([idx, boxes], dim=1)


def find_float_boundary(maskdt, width):
    # Turn hard masks (0 or 1) into a "soft" boundary map.
    # Think: where the mask changes from 0->1 or 1->0 is the boundary.
    # This returns a float map (same shape) that is larger near edges.
    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, stride=1, padding=width // 2)
    # bml measures distance from full patch (interior), bms from zero (exterior)
    bml = torch.abs(boundary_mask - width * width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width * width / 2)
    return fbmask.view(N, H, W)


def get_dets(fbmask, patch_size, iou_thresh):
    # From the soft boundary map, create candidate square patches centered on
    # boundary pixels. Use NMS to remove overlapping boxes. The result is a
    # set of boxes with a score (how strong the boundary was there).
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys, xs]
    ys = ys.float()
    xs = xs.float()
    dets = torch.stack([xs - patch_size // 2, ys - patch_size // 2,
                        xs + patch_size // 2, ys + patch_size // 2, scores]).T
    _, inds = nms(dets[:, :4].contiguous(), dets[:, 4].contiguous(), iou_thresh)
    sdets = dets[inds]
    H, W = fbmask.shape
    return _force_move_back(sdets, H, W, patch_size)


def boxes_to_clockwise_corners(boxes):
    """
    Convert axis-aligned boxes [x1,y1,x2,y2] into corner coordinates
    ordered clockwise starting at top-left: (x1,y1),(x2,y1),(x2,y2),(x1,y2).

    Input: boxes tensor shape (N,4)
    Output: tensor shape (N,8) with ordering [x1,y1,x2,y1,x2,y2,x1,y2]
    """
    if boxes.numel() == 0:
        return boxes.new_empty((0, 8)).long()

    # Ensure x1<x2 and y1<y2 even if boxes are unordered
    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])

    # Clockwise ordering starting at top-left: TL, TR, BR, BL
    corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=1)

    # Round to nearest integer pixel coordinates and return as long (int64)
    corners = torch.round(corners).long()
    return corners


# ========================
# Split function
# ========================
def split(img, gt, pred_t, boundary_width=3, iou_thresh=0.55, patch_size=64, out_size=64):
    """
    Extract patches around mask boundaries.

    Inputs:
      - img: HxWx3 numpy image (RGB)
      - maskdts: tensor (K,H,W) of binary masks (float 0/1)
      - pred_t: tensor (K,H,W) of predicted masks (float 0/1) or similar
    Returns:
      - detss: list of tensors of detections for each instance
      - img_patches: tensor (N, C, patch_size, patch_size)
      - dt_patches: tensor (N, 1, patch_size, patch_size) (ground-truth crops)
      - pred_patches: tensor (N, 1, patch_size, patch_size) (prediction crops)

        Notes:
            - If no detections are found returns empty list and empty tensors.
    """
    # 1) Convert each predicted instance mask into a soft boundary map.
    #    We use predicted masks (pred_t) to generate proposals as requested.
    fbmasks = find_float_boundary(pred_t, boundary_width)
    # 2) For each instance, find patch boxes around strong boundary pixels.
    detss = []
    for i in range(fbmasks.size(0)):
        # dets shape: (M,5) -> keep first 4 cols (x1,y1,x2,y2)
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:, :4]
        detss.append(dets)

    # If no boundary boxes were found, return empty tensors so the caller can
    # skip this image without errors.
    if len(detss) == 0 or all([d.size(0) == 0 for d in detss]):
        print(detss)
        return [], torch.empty(0), torch.empty(0), torch.empty(0)

    all_dets = torch.cat(detss, dim=0)

    # For each detection compute the 4 corner coordinates in clockwise order.
    # det_corners_list mirrors detss (list per instance), where each element
    # is a tensor shape (num_boxes, 8) containing [x1,y1,x2,y1,x2,y2,x1,y2].
    # det_corners_list = [boxes_to_clockwise_corners(d) for d in detss]

    # 3) Crop RGB image patches using roi_align. roi_align needs tensors in
    # (B,C,H,W) format. We have one image so batch size is 1.
    img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float().contiguous()
    img_patches = roi_align(img_t, _to_rois(all_dets), patch_size)

    _detss = [torch.cat([i * _.new_ones((_.size(0), 1)), _], dim=1) for i, _ in enumerate(detss)]
    _detss = torch.cat(_detss)
    # 4) Crop corresponding ground-truth and predicted mask patches. These are
    # single-channel (1,H,W) so we add a channel dim before roi_align.
    gt_patches = roi_align(gt[:, None, :, :], _detss, patch_size)
    pred_patches = roi_align(pred_t[:, None, :, :], _detss, patch_size)

    # Return values (easy terms):
    #  - detss: list of box tensors, one list per ground-truth instance.
    #  - img_patches: RGB patch images ready for saving or further model input.
    #  - dt_patches: the corresponding ground-truth mask crops (useful for
    #                training or evaluation).
    #  - pred_patches: the predicted mask crops (what the coarse model predicted).
    # Return the original detss plus the clockwise corner coordinates as a
    # supplementary result (det_corners_list). The function now returns:
    # detss, img_patches, dt_patches, pred_patches, det_corners_list
    return detss, img_patches, gt_patches, pred_patches


# ========================
# Main loop
# ========================

# Input paths
type = 'train'
root_dir = '../dataset/pranet-traindataset/pranet-testdataset'
output_dir = '../dataset/pranet-traindataset/testPatchesDataset'

img_path = os.path.join(root_dir, "images")
gt_path = os.path.join(root_dir, "gts")
mask_path = os.path.join(root_dir, "masks")

file_list = [f for f in os.listdir(gt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


out_img_dir = os.path.join(output_dir, "img_dir", type)
out_gt_dir = os.path.join(output_dir, "ann_dir", type)
out_pred_dir = os.path.join(output_dir, "mask_dir", type)
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_gt_dir, exist_ok=True)
os.makedirs(out_pred_dir, exist_ok=True)
print(len(file_list))
# import random
# selected_files = random.sample(file_list, 644)
for filename in file_list:
    base = os.path.splitext(filename)[0]
    img_file = os.path.join(img_path, filename)
    gt_file = os.path.join(gt_path, filename)
    pred_file = os.path.join(mask_path, filename)

    if not os.path.exists(img_file) or not os.path.exists(pred_file) or not os.path.exists(gt_file):
        print(f"Skipping {filename}: missing image or prediction file.")
        continue

    img = cv2.imread(img_file)[:, :, ::-1].copy()  # BGR -> RGB
    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

    gt_t = torch.from_numpy(gt / 255.0).float().unsqueeze(0)  # (1,H,W)
    pred_t = torch.from_numpy(pred / 255.0).float().unsqueeze(0)
    # if calculate_iou(pred_t, gt_t) < 0.5:
    #     print(f"Skipping {filename}")
    #     continue

    detss, img_patches, gt_patches, pred_patches = split(img, gt_t, pred_t)
    if detss == []:
        print(f"no patches: {filename}")
        continue

    TEMP_SAVE_PATCHES = True
    if TEMP_SAVE_PATCHES:
        img_patches_np = img_patches.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for idx in range(img_patches_np.shape[0]):
            cv2.imwrite(
                os.path.join(out_img_dir, f"{base}_patch_{idx:04d}.png"),
                img_patches_np[idx][:, :, ::-1]  # RGB -> BGR
            )

        gt_patches_np = (gt_patches.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
        for idx in range(gt_patches_np.shape[0]):
            cv2.imwrite(
                os.path.join(out_gt_dir, f"{base}_patch_{idx:04d}.png"),
                gt_patches_np[idx]
            )

        pred_patches_np = (pred_patches.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
        for idx in range(pred_patches_np.shape[0]):
            cv2.imwrite(
                os.path.join(out_pred_dir, f"{base}_patch_{idx:04d}.png"),
                pred_patches_np[idx]
            )


print("\n All images processed using ROI-based patch extraction.")