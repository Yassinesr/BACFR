"""TTA-variance-gated merge sweep — alternative to the dual-head version.

Motivation
----------
Test_patch_tta_oracle.py showed the dual-head uncertainty gating
captures only ~+0.001 of the ~+0.013 pixel-wise oracle headroom on the
polyppvt->Polyp-PVT recipe. The dual heads were trained to predict
*model confidence* (uncertainty-weighted BCE during training), which is
not the same as correctness — especially on a strong base where the
model can be confidently wrong on a few pixels.

This script gates with a different signal: **TTA variance** across the
4 flip views. With flip-consistency training (FCT), the model is
trained to be flip-equivariant. High variance across the 4 views =
the model is failing to be equivariant here = unreliable. Low variance
= the model agrees with itself across orientations = reliable.

Same gate formula as the uncertainty version:
    use_refiner where signal_norm <= tau, else keep coarse.
Self-validation anchors are identical:
    tau = 1.0 -> reproduces current merge (~0.8726).
    tau = 0.0 -> reproduces raw coarse base (~0.8732 on this server).

GROUND TRUTH IS USED ONLY FOR SCORING. Never fed to the model.
Model input is strictly (RGB image patch, coarse-mask patch).
"""
import os
import os.path as osp
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from utils.utils import parse_args, load_config, to_cuda
from utils.dataloader import *
from lib import *
from run.Test_patch_tta import split

PATCH_SIZE = 64


# ==============================================================
# TTA returning (mean prediction, variance across 4 views)
# ==============================================================
@torch.no_grad()
def tta_pred_var(model, img_patches, dt_patches):
    """Returns mean over 4 flip views and per-pixel variance across them.
    Both un-flipped to canonical orientation."""
    views = [None, [3], [2], [2, 3]]
    preds = []
    for fd in views:
        if fd is None:
            s = {'image': img_patches, 'mask': dt_patches}
        else:
            s = {'image': torch.flip(img_patches, dims=fd),
                 'mask':  torch.flip(dt_patches, dims=fd)}
        out = model(to_cuda(s))
        p = torch.sigmoid(out['pred'])
        if fd is not None:
            p = torch.flip(p, dims=fd)
        preds.append(p)
    stack = torch.stack(preds, dim=0)            # (4, B, 1, H, W)
    pred_mean = stack.mean(dim=0)                # (B, 1, H, W)
    pred_var = stack.var(dim=0, unbiased=False)  # (B, 1, H, W) -- in [0, 0.25]
    return pred_mean, pred_var


# ==============================================================
# Gated merge — identical to the uncertainty version but takes a
# generic signal array. Higher signal = less reliable = prefer coarse.
# ==============================================================
def merge_gated(coarse_bool, dets, pred_ds, sig_ds, tau, patch_size=PATCH_SIZE):
    H, W = coarse_bool.shape
    out = coarse_bool.copy()
    acc = np.zeros((H, W), np.float32)
    accs = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in dets[i][:4]]
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            continue
        pr = pred_ds[i]
        sg = sig_ds[i]
        if (h, w) != (patch_size, patch_size):
            pr = np.array(Image.fromarray(pr).resize((w, h), Image.BILINEAR))
            sg = np.array(Image.fromarray(sg).resize((w, h), Image.BILINEAR))
        acc[y1:y2, x1:x2] += pr
        accs[y1:y2, x1:x2] += sg
        cnt[y1:y2, x1:x2] += 1

    m = cnt > 0
    if not m.any():
        return out
    acc[m] /= cnt[m]
    accs[m] /= cnt[m]

    refined_bin = acc[m] > 0.5
    s = accs[m]
    if s.max() - s.min() < 1e-8:
        out[m] = refined_bin
        return out
    s_norm = (s - s.min()) / (s.max() - s.min())
    use_refiner = s_norm <= tau

    region = out[m].copy()
    region[use_refiner] = refined_bin[use_refiner]
    out[m] = region
    return out


def dice_bin(pred_bool, gt_bool):
    num_and = np.logical_and(pred_bool, gt_bool).sum()
    num_obj = gt_bool.sum()
    num_pred = pred_bool.sum()
    if num_and == 0:
        return 0.0
    return float(2 * num_and / (num_obj + num_pred))


def _load_gt(gt_root, name, shape_hw):
    p = osp.join(gt_root, name)
    if not osp.isfile(p):
        p = osp.join(gt_root, osp.splitext(name)[0] + '.png')
    g = Image.open(p).convert('L')
    if g.size[::-1] != shape_hw:
        g = g.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
    return np.array(g).astype(np.float32) / 255.0 > 0.5


if __name__ == '__main__':
    args = parse_args()
    config = args.config if os.path.isfile(args.config) else 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[var-gated] using config: {config}')
    opt = load_config(config)

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pth', type=str, default=None)
    extra.add_argument('--dt_path', type=str, required=True)
    extra.add_argument('--img_root', type=str,
                       default='/home/yassine/projects/UACANet-main/dataset/TestDataset')
    extra.add_argument('--taus', type=str,
                       default='1.0,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0')
    ex, _ = extra.parse_known_args()
    taus = [float(t) for t in ex.taus.split(',')]

    pth = ex.pth or osp.join(opt.Test.Checkpoint.checkpoint_dir, 'best.pth')
    print(f'[var-gated] checkpoint: {pth}')
    print(f'[var-gated] coarse masks: {ex.dt_path}')
    print(f'[var-gated] signal: TTA variance across 4 flip views')

    model = eval(opt.Model.name)(
        channels=opt.Model.channels, output_stride=opt.Model.output_stride,
        pretrained=opt.Model.pretrained,
        use_mccpb=getattr(opt.Model, 'use_mccpb', False),
        use_dual_heads=getattr(opt.Model, 'use_dual_heads', False),
        use_boundary_contrast=getattr(opt.Model, 'use_boundary_contrast', False),
        use_hf_gate=getattr(opt.Model, 'use_hf_gate', False),
        use_flip_consistency=getattr(opt.Model, 'use_flip_consistency', False),
        edge_dist_mode=getattr(opt.Model, 'edge_dist_mode', 'cdist'))
    ckpt = torch.load(pth, map_location='cuda')
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    state = state.get('state_dict', state) if isinstance(state, dict) and 'state_dict' in state else state
    model.load_state_dict(state, strict=True)
    model.cuda().eval()

    cache = {ts: [] for ts in opt.Test.Dataset.datasets}

    for testset in opt.Test.Dataset.datasets:
        img_path = osp.join(ex.img_root, testset, 'images')
        gt_path = osp.join(ex.img_root, testset, 'gts')
        mask_path = osp.join(ex.dt_path, testset)

        ds = eval(opt.Test.Dataset.type)(
            img_root=img_path, mask_root=mask_path,
            transform_list=opt.Test.Dataset.transform_list)
        loader = data.DataLoader(ds, batch_size=1,
                                 num_workers=opt.Test.Dataloader.num_workers,
                                 pin_memory=opt.Test.Dataloader.pin_memory)

        for sample in loader:
            coarse = sample['gt'].squeeze(1)
            image = sample['image']
            name = sample['name'][0]
            H, W = coarse.shape[-2], coarse.shape[-1]
            coarse_bool = coarse.squeeze(0).cpu().numpy() > 0.5
            gt_bool = _load_gt(gt_path, name, (H, W))

            dets, img_patches, dt_patches = split(image, coarse)
            if dets is None:
                cache[testset].append((coarse_bool, gt_bool,
                                       np.zeros((0, 4)),
                                       np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32),
                                       np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32)))
                continue

            preds, vars_ = [], []
            for i in range(0, len(img_patches), 8):
                pm, pv = tta_pred_var(model, img_patches[i:i + 8], dt_patches[i:i + 8])
                pm = F.interpolate(pm, (PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
                pv = F.interpolate(pv, (PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
                preds.append(pm.squeeze(1).cpu().numpy())
                vars_.append(pv.squeeze(1).cpu().numpy())
            pred_ds = np.concatenate(preds, 0).astype(np.float32)
            var_ds = np.concatenate(vars_, 0).astype(np.float32)
            dets_np = (torch.cat(dets, 0).cpu().numpy()
                       if isinstance(dets, list) else dets.cpu().numpy())
            cache[testset].append((coarse_bool, gt_bool, dets_np, pred_ds, var_ds))
        print(f'[var-gated] cached {testset}: {len(cache[testset])} images')

    # ---- sweep ----
    print('\n' + '=' * 92)
    header = f'{"tau":>6} | ' + ' | '.join(f'{ts[:12]:>12}' for ts in opt.Test.Dataset.datasets) + ' | ' + f'{"MEAN":>7}'
    print(header)
    print('-' * len(header))
    for tau in taus:
        per_set_means = []
        for testset in opt.Test.Dataset.datasets:
            dices = []
            for (coarse_bool, gt_bool, dets_np, pred_ds, var_ds) in cache[testset]:
                if dets_np.shape[0] == 0:
                    pred_bool = coarse_bool
                else:
                    pred_bool = merge_gated(coarse_bool, dets_np, pred_ds, var_ds, tau)
                dices.append(dice_bin(pred_bool, gt_bool))
            per_set_means.append(float(np.mean(dices)))
        mean = float(np.mean(per_set_means))
        row = f'{tau:>6.2f} | ' + ' | '.join(f'{v:>12.4f}' for v in per_set_means) + ' | ' + f'{mean:>7.4f}'
        print(row)
    print('=' * 92)
    print('Anchors: tau=1.00 should match the current run (~0.8726);')
    print('         tau=0.00 should match the raw coarse base (~0.8732).')
    print()
    print('For comparison, prior sweeps on this recipe:')
    print('  dual-head uncertainty gating peaked at +0.001 above current.')
    print('  oracle pixel-wise upper bound  : +0.0128 above current.')
    print('If TTA variance peaks meaningfully higher than +0.001, we have a')
    print('better signal. If it also caps at ~+0.001, no model-internal signal')
    print('on this checkpoint will work -- only mixed training (different')
    print('refiner) can move the needle.')
