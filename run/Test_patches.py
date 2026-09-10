import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.nn import modules

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from utils.utils import *
from utils.dataloader import *
from utils.utils import parse_args, load_config, to_cuda
import torch.utils.data as data
from lib import *
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import mmcv
from torchvision.ops import nms
from torchvision.ops import roi_align
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader


# ======================================================
# Build model with v3.x flags pulled from config (with safe defaults)
# ======================================================
def build_model_from_config(opt):
    """Instantiate the model with all v3.x flags from the YAML config.
    Falls back to safe defaults (False) for any missing flag, so this
    also works for the original BACFR class which ignores **kwargs.
    """
    name = opt.Model.name
    kwargs = dict(
        channels=opt.Model.channels,
        output_stride=opt.Model.output_stride,
        pretrained=opt.Model.pretrained,
    )
    # v3.x toggle flags — only pass when the config has them
    optional_flags = [
        'use_mccpb',
        'use_dual_heads',
        'use_boundary_contrast',
        'use_hf_gate',
        'use_edge_gate',
        'edge_dist_mode',
        'attn_hidden',
        # UACANet_Refine / _FCT
        'guidance_scale',
        'use_flip_consistency',
        'fct_weight',
        'fct_warmup_iters',
        'fct_use_vflip',
        'fct_supervise_flips',
    ]
    for flag in optional_flags:
        if hasattr(opt.Model, flag):
            kwargs[flag] = getattr(opt.Model, flag)

    model_class = eval(name)
    # Probe the constructor signature so older classes (BACFR baseline)
    # don't choke on flags they don't accept.
    import inspect
    sig = inspect.signature(model_class.__init__)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        print(f"[build_model_from_config] dropped (not in {name}.__init__): {sorted(dropped)}")
    return model_class(**filtered)


# ======================================================
# Collect test dataset images and masks
# ======================================================
def collect_test_paths(root, pred_root):
    img_paths, dt_paths = [], []
    for dataset_name in sorted(os.listdir(root)):
        dataset_dir = osp.join(root, dataset_name)
        images_dir = osp.join(dataset_dir, "images")
        preds_dir = osp.join(pred_root, dataset_name)

        if not osp.isdir(images_dir) or not osp.isdir(preds_dir):
            continue

        for img_name in sorted(os.listdir(images_dir)):
            img_path = osp.join(images_dir, img_name)
            mask_path = osp.join(preds_dir, img_name)
            if osp.exists(mask_path):
                img_paths.append(img_path)
                dt_paths.append([mask_path])
    return img_paths, dt_paths


# ======================================================
# Boundary & Patch Functions
# ======================================================
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


def get_dets(fbmask, patch_size, iou_thresh=0.3):
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys, xs]
    ys = ys.float()
    xs = xs.float()
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
        print("Warning: no boundary detected, returning None")
        return None, None, None

    img = img.float().contiguous()
    img_patches = roi_align(img, _to_rois(all_dets), patch_size, aligned=True)

    _detss = [torch.cat([i * _.new_ones((_.size(0), 1)), _], dim=1) for i, _ in enumerate(detss)]
    _detss = torch.cat(_detss)
    dt_patches = roi_align(maskdts[:, None, :, :], _detss, patch_size, aligned=True)

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

        dt_refined.zero_()
        dt_count.zero_()
        for i in range(dets.size(0)):
            x1, y1, x2, y2 = dets[i]
            dt_refined[y1:y2, x1:x2] += masks[i]
            dt_count[y1:y2, x1:x2] += 1

        s = dt_count > 0
        dt_refined[s] /= dt_count[s]

        maskdt[s] = dt_refined[s] > 0.5
        out.append(maskdt)
    return out


# ======================================================
# Inference Loop
# ======================================================
def test(opt, args, out_dir, pth, type, dt_path, model):
    os.makedirs(out_dir, exist_ok=True)
    # Patch model-input resolution; must match the training resize.
    out_size = int(getattr(opt.Test.Dataset, 'out_size', 256))

    state = torch.load(pth, map_location='cpu')
    # Support either raw state_dict or {'state_dict': ..., 'config': ...} bundle
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    msg = model.load_state_dict(state_dict, strict=True)
    if hasattr(msg, 'missing_keys') and (msg.missing_keys or msg.unexpected_keys):
        print("Missing keys:   ", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

    model.cuda()
    model.eval()

    for testset in opt.Test.Dataset.datasets:
        save_dir = os.path.join(out_dir, testset)
        os.makedirs(save_dir, exist_ok=True)

        root = opt.Test.Dataset.root
        img_subdir = getattr(opt.Test.Dataset, 'img_subdir', 'images')
        img_path = os.path.join(root, testset, img_subdir)
        mask_path = os.path.join(dt_path, testset)
        test_dataset = eval(opt.Test.Dataset.type)(
            img_root=img_path, mask_root=mask_path,
            transform_list=opt.Test.Dataset.transform_list,
        )

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=opt.Test.Dataloader.num_workers,
            pin_memory=opt.Test.Dataloader.pin_memory,
        )

        for sample in test_loader:
            mask = sample['gt'].squeeze(1)
            image = sample['image']

            s = {}
            dets, img_patches, dt_patches = split(image, mask, out_size=out_size)

            if dets is None:
                print(sample['name'])
                continue

            refinemasks_final = []
            for i in range(0, len(img_patches), 8):
                s['image'] = img_patches[i:i + 8]
                s['mask'] = dt_patches[i:i + 8]
                s = to_cuda(s)
                refine = model(s)
                pred = refine['pred'].squeeze(1)
                pred = torch.sigmoid(pred)
                refinemasks_final += pred.tolist()

            refinemasks_final = torch.tensor(refinemasks_final).cuda(device=torch.device('cuda:0'))

            refineds = merge((mask.cuda(device=torch.device('cuda:0'))), dets, refinemasks_final)
            for i in range(len(refineds)):
                cv2.imwrite(
                    osp.join(save_dir, sample['name'][0]),
                    refineds[i].cpu().numpy().astype(np.uint8) * 255,
                )


# ======================================================
# Main
# ======================================================
if __name__ == '__main__':
    import argparse
    args = parse_args()

    # CLI overrides (config comes from --config via parse_args; these layer on top).
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pth', type=str, default=None,
                       help='checkpoint .pth; default: <checkpoint_dir>/latest.pth')
    extra.add_argument('--out_dir', type=str, default=None,
                       help='output dir; default: results_cl/<ckpt_dir_basename>_noTTA')
    extra.add_argument('--dt_path', type=str, default=None,
                       help='coarse-mask source root (per-testset subdirs)')
    ex, _ = extra.parse_known_args()

    config = args.config if os.path.isfile(args.config) else 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[Test_patches] using config: {config}')
    opt = load_config(config)

    ckpt_dir = opt.Test.Checkpoint.checkpoint_dir
    pth = ex.pth or os.path.join(ckpt_dir, 'latest.pth')
    out_dir = ex.out_dir or os.path.join(
        'results_cl', os.path.basename(ckpt_dir.rstrip('/')) + '_noTTA')
    dt_path = (ex.dt_path
               or getattr(opt.Test.Dataset, 'dt_path', None)
               or os.environ.get('BACFR_DT_PATH'))
    if not dt_path:
        raise ValueError('No coarse-mask source: pass --dt_path or set '
                         'Test.Dataset.dt_path in the config.')

    # build_model_from_config filters kwargs via inspect, so plain models
    # (UACANet: channels/output_stride/pretrained only) build fine.
    model = build_model_from_config(opt)

    print(out_dir)
    print(pth)
    print("Model:", opt.Model.name)
    print(f"coarse-mask source (dt_path): {dt_path}")

    test(opt, args, out_dir, pth, None, dt_path, model)
