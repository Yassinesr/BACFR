"""Test patch refinement with Test-Time Augmentation.

Runs each patch through 4 flip combinations and averages sigmoid outputs.
Identical to Test_patch.py except for the model() call which is wrapped
in tta_predict().
"""
import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2

from torchvision.ops import nms, roi_align

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from utils.utils import parse_args, load_config, to_cuda
from utils.dataloader import *
from lib import *


# ==============================================================
# TTA wrapper — the only new piece vs Test_patch.py
# ==============================================================
@torch.no_grad()
def tta_predict(model, sample):
    """Run 4 flip variants, average sigmoid outputs.

    For polyp masks the 4 D4 sub-group transforms that preserve label
    semantics are: identity, H-flip, V-flip, H+V-flip. We apply each
    to BOTH image and coarse mask, run the model, un-flip the output,
    then average.
    """
    img = sample['image']
    mask = sample['mask']

    # Identity
    s = to_cuda({'image': img, 'mask': mask})
    p0 = torch.sigmoid(model(s)['pred'])

    # Horizontal flip (dim=3 in NCHW)
    s = to_cuda({'image': torch.flip(img, dims=[3]),
                 'mask':  torch.flip(mask, dims=[3])})
    p1 = torch.flip(torch.sigmoid(model(s)['pred']), dims=[3])

    # Vertical flip (dim=2)
    s = to_cuda({'image': torch.flip(img, dims=[2]),
                 'mask':  torch.flip(mask, dims=[2])})
    p2 = torch.flip(torch.sigmoid(model(s)['pred']), dims=[2])

    # Both flips (180° rotation)
    s = to_cuda({'image': torch.flip(img, dims=[2, 3]),
                 'mask':  torch.flip(mask, dims=[2, 3])})
    p3 = torch.flip(torch.sigmoid(model(s)['pred']), dims=[2, 3])

    return (p0 + p1 + p2 + p3) / 4.0


# ==============================================================
# Patch helpers (verbatim from Test_patch.py)
# ==============================================================
def find_float_boundary(maskdt, width):
    if maskdt.ndim == 2:
        maskdt = maskdt.unsqueeze(0)
    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, stride=1, padding=width // 2)
    bml = torch.abs(boundary_mask - width * width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width * width / 2)
    return fbmask.view(N, H, W)


def _force_move_back(sdets, H, W, patch_size):
    s = sdets[:, 0] < 0; sdets[s, 0] = 0; sdets[s, 2] = patch_size
    s = sdets[:, 1] < 0; sdets[s, 1] = 0; sdets[s, 3] = patch_size
    s = sdets[:, 2] >= W; sdets[s, 0] = W - 1 - patch_size; sdets[s, 2] = W - 1
    s = sdets[:, 3] >= H; sdets[s, 1] = H - 1 - patch_size; sdets[s, 3] = H - 1
    return sdets


def get_dets(fbmask, patch_size, iou_thresh=0.3):
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys, xs]
    ys = ys.float(); xs = xs.float()
    dets = torch.stack([xs - patch_size // 2, ys - patch_size // 2,
                        xs + patch_size // 2, ys + patch_size // 2, scores]).T
    inds = nms(dets[:, :4].contiguous(), dets[:, 4].contiguous(), iou_thresh)
    sdets = dets[inds]
    H, W = fbmask.shape
    return _force_move_back(sdets, H, W, patch_size)


def _to_rois(xyxys):
    inds = xyxys.new_zeros((xyxys.size(0), 1))
    return torch.cat([inds, xyxys], dim=1).float().contiguous()


def split(img, maskdts, boundary_width=3, iou_thresh=0.55, patch_size=64, out_size=256):
    fbmasks = find_float_boundary(maskdts, boundary_width)
    detss = []
    for i in range(fbmasks.size(0)):
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:, :4]
        detss.append(dets)
    all_dets = torch.cat(detss, dim=0)
    if all_dets.size(0) == 0:
        return None, None, None
    img = img.float().contiguous()
    img_patches = roi_align(img, _to_rois(all_dets), patch_size)
    _detss = [torch.cat([i * _.new_ones((_.size(0), 1)), _], dim=1) for i, _ in enumerate(detss)]
    _detss = torch.cat(_detss)
    dt_patches = roi_align(maskdts[:, None, :, :], _detss, patch_size)
    img_patches = F.interpolate(img_patches, (out_size, out_size), mode='bilinear')
    dt_patches = F.interpolate(dt_patches, (out_size, out_size), mode='nearest')
    return detss, img_patches, dt_patches


def merge(maskdts, detss, maskss, patch_size=64):
    out = []
    K, H, W = maskdts.shape
    maskdts = maskdts.bool()
    maskss = F.interpolate(maskss.unsqueeze(0), (patch_size, patch_size), mode='bilinear').squeeze(0)
    dt_refined = torch.zeros_like(maskdts[0], dtype=torch.float32)
    dt_count = torch.zeros_like(maskdts[0], dtype=torch.float32)
    p = 0
    for k in range(K):
        dets = detss[k][:, :4].int()
        maskdt = maskdts[k]
        q = p + dets.size(0)
        masks = maskss[p:q]
        p = q
        dt_refined.zero_(); dt_count.zero_()
        for i in range(dets.size(0)):
            x1, y1, x2, y2 = dets[i]
            dt_refined[y1:y2, x1:x2] += masks[i]
            dt_count[y1:y2, x1:x2] += 1
        s = dt_count > 0
        dt_refined[s] /= dt_count[s]
        maskdt[s] = dt_refined[s] > 0.5
        out.append(maskdt)
    return out


# ==============================================================
# Test loop (only change vs Test_patch.py: tta_predict instead of model())
# ==============================================================
def test(opt, args, out_dir, pth, dt_path):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(pth, map_location='cuda')

    # Handle different checkpoint save formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    	state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    	state_dict = ckpt['state_dict']
    else:
    	state_dict = ckpt  # assume it's already a bare state dict

    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    for testset in opt.Test.Dataset.datasets:
        save_dir = os.path.join(out_dir, testset)
        os.makedirs(save_dir, exist_ok=True)

        root = "/home/yassine/projects/UACANet-main/dataset/TestDataset"
        img_path = os.path.join(root, testset, 'gts')
        mask_path = os.path.join(dt_path, testset)
        test_dataset = eval(opt.Test.Dataset.type)(
            img_root=img_path, mask_root=mask_path,
            transform_list=opt.Test.Dataset.transform_list)
        test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=1,
            num_workers=opt.Test.Dataloader.num_workers,
            pin_memory=opt.Test.Dataloader.pin_memory)

        for sample in test_loader:
            mask = sample['gt'].squeeze(1)
            image = sample['image']
            dets, img_patches, dt_patches = split(image, mask)
            if dets is None:
                print(sample['name'])
                continue

            refinemasks_final = []
            for i in range(0, len(img_patches), 8):
                # >>> THE TTA CHANGE: replace single forward with 4-flip average <
                s = {'image': img_patches[i:i + 8],
                     'mask':  dt_patches[i:i + 8]}
                pred = tta_predict(model, s)        # (B, 1, 256, 256), already sigmoid
                pred = pred.squeeze(1)
                refinemasks_final += pred.tolist()
                # >>> END TTA CHANGE <

            refinemasks_final = torch.tensor(refinemasks_final).cuda()
            refineds = merge(mask.cuda(), dets, refinemasks_final)
            for i in range(len(refineds)):
                cv2.imwrite(osp.join(save_dir, sample['name'][0]),
                            refineds[i].cpu().numpy().astype(np.uint8) * 255)


if __name__ == '__main__':
    args = parse_args()
    config = 'configs/BACFR_Enhanced_v3_3.yaml'
    opt = load_config(config)

    # Change these two paths to point at your epoch-2 checkpoint and desired output
    pth = 'checkpoints/BACFR_Enhanced_v3_3/epoch_4.pth'
    out_dir = 'results_cl/BACFR_FCT_TTA'
    dt_path = "/home/yassine/projects/UACANet-main/results_cl/paper_results/PraNet-results/PraNet"

    model = eval(opt.Model.name)(
        channels=opt.Model.channels,
        output_stride=opt.Model.output_stride,
        pretrained=opt.Model.pretrained,
        use_mccpb=getattr(opt.Model, 'use_mccpb', False),
        use_dual_heads=getattr(opt.Model, 'use_dual_heads', False),
        use_boundary_contrast=getattr(opt.Model, 'use_boundary_contrast', False),
        use_hf_gate=getattr(opt.Model, 'use_hf_gate', False),
        edge_dist_mode=getattr(opt.Model, 'edge_dist_mode', 'cdist'),
    )

    print(f"Running TTA inference: {pth} -> {out_dir}")
    test(opt, args, out_dir, pth, dt_path)
    print("Done. Now run your eval script on this folder.")
