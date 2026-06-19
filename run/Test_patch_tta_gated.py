"""Dual-head uncertainty-gated merge — inference-only sweep.

Motivation
----------
On a strong base segmenter (Polyp-PVT, ~0.870 raw), the BACFR refiner
adds almost nothing (+0.002) because the original merge() does a HARD
replace: every boundary patch overwrites the coarse mask, including
where the refiner is unsure and the coarse mask was already right. That
over-correction cancels most of the genuine gains.

This script uses the refiner's OWN dual-head uncertainty
    u = sigmoid(fg_pred) * sigmoid(bg_pred)
(currently computed at inference and then discarded) to GATE the merge:
keep the coarse Polyp-PVT mask where the refiner is uncertain; use the
refined prediction only where it is confident.

The model runs ONCE per image; the refined + uncertainty patches are
cached (downsampled to patch_size), then we sweep the gate threshold
`tau` offline in seconds and score each setting against GT.

Self-validation anchors (printed first):
  * tau = 1.0  -> use refiner everywhere touched  == current merge.
                 Mean Dice should reproduce the known 0.8726 run.
  * tau = 0.0  -> never use refiner                == raw coarse mask.
                 Mean Dice should reproduce raw Polyp-PVT (~0.870).
If either anchor is off, the GT-scoring harness has a bug — stop and fix
before trusting any intermediate tau.

GROUND TRUTH IS USED ONLY FOR SCORING. It is never fed to the model.
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
from run.Test_patch_tta import split  # reuse exact patch-cropping logic


PATCH_SIZE = 64  # must match merge() / split() in Test_patch_tta.py


# ==============================================================
# TTA that also returns the dual-head uncertainty
# ==============================================================
@torch.no_grad()
def tta_predict_unc(model, img_patches, dt_patches):
    """4-view flip TTA. Returns (pred_prob, uncertainty), each averaged
    over the 4 views and un-flipped back to canonical orientation."""
    views = [None, [3], [2], [2, 3]]
    p_sum = None
    u_sum = None
    for fd in views:
        if fd is None:
            s = {'image': img_patches, 'mask': dt_patches}
        else:
            s = {'image': torch.flip(img_patches, dims=fd),
                 'mask':  torch.flip(dt_patches, dims=fd)}
        out = model(to_cuda(s))
        p = torch.sigmoid(out['pred'])
        # Dual-head uncertainty. If dual heads are disabled the model
        # returns zeros -> u == 0.25 everywhere (degenerate); we detect
        # and warn at merge time.
        fg = torch.sigmoid(out['fg_pred'])
        bg = torch.sigmoid(out['bg_pred'])
        u = fg * bg
        if fd is not None:
            p = torch.flip(p, dims=fd)
            u = torch.flip(u, dims=fd)
        p_sum = p if p_sum is None else p_sum + p
        u_sum = u if u_sum is None else u_sum + u
    return p_sum / 4.0, u_sum / 4.0


# ==============================================================
# Gated merge (operates on cached, patch_size-downsampled patches)
# ==============================================================
def merge_gated(coarse_bool, dets, pred_ds, unc_ds, tau, patch_size=PATCH_SIZE):
    """coarse_bool : (H,W) bool — the coarse mask to be selectively refined.
       dets        : (N,4) int xyxy boxes in full-image coords.
       pred_ds     : (N,ps,ps) float — refined sigmoid prob per patch.
       unc_ds      : (N,ps,ps) float — dual-head uncertainty per patch.
       tau         : keep coarse where normalized uncertainty > tau;
                     use refiner where <= tau. tau=1 -> always refiner
                     (current behavior). tau=0 -> never refiner (raw).
    """
    H, W = coarse_bool.shape
    out = coarse_bool.copy()
    acc = np.zeros((H, W), np.float32)
    accu = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in dets[i][:4]]
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            continue
        pr = pred_ds[i]
        un = unc_ds[i]
        if (h, w) != (patch_size, patch_size):
            # box clipped at image edge — resize the patch to fit
            pr = np.array(Image.fromarray(pr).resize((w, h), Image.BILINEAR))
            un = np.array(Image.fromarray(un).resize((w, h), Image.BILINEAR))
        acc[y1:y2, x1:x2] += pr
        accu[y1:y2, x1:x2] += un
        cnt[y1:y2, x1:x2] += 1

    m = cnt > 0
    if not m.any():
        return out
    acc[m] /= cnt[m]
    accu[m] /= cnt[m]

    refined_bin = acc[m] > 0.5
    u = accu[m]
    if u.max() - u.min() < 1e-8:
        # degenerate uncertainty (dual heads off?) -> fall back to hard replace
        out[m] = refined_bin
        return out
    u_norm = (u - u.min()) / (u.max() - u.min())
    use_refiner = u_norm <= tau   # confident enough to trust the refiner

    region_vals = out[m].copy()
    region_vals[use_refiner] = refined_bin[use_refiner]
    out[m] = region_vals
    return out


# ==============================================================
# Dice (binary), matching utils/eval_functions.Fmeasure_calu
# ==============================================================
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
        alt = osp.splitext(name)[0] + '.png'
        p = osp.join(gt_root, alt)
    g = Image.open(p).convert('L')
    if g.size[::-1] != shape_hw:
        g = g.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
    g = np.array(g).astype(np.float32) / 255.0
    return g > 0.5


# ==============================================================
# Main
# ==============================================================
if __name__ == '__main__':
    args = parse_args()
    config = args.config if os.path.isfile(args.config) else 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[gated] using config: {config}')
    opt = load_config(config)

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pth', type=str, default=None)
    extra.add_argument('--dt_path', type=str, required=True,
                       help='coarse-mask source root (per-testset subdirs)')
    extra.add_argument('--img_root', type=str,
                       default='/home/yassine/projects/UACANet-main/dataset/TestDataset',
                       help='TestDataset root with <set>/images and <set>/gts')
    extra.add_argument('--taus', type=str, default='1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.0',
                       help='comma-separated gate thresholds to sweep')
    ex, _ = extra.parse_known_args()

    taus = [float(t) for t in ex.taus.split(',')]
    pth = ex.pth
    if pth is None:
        ckpt_dir = opt.Test.Checkpoint.checkpoint_dir
        pth = osp.join(ckpt_dir, 'best.pth')
    print(f'[gated] checkpoint: {pth}')
    print(f'[gated] coarse masks (dt_path): {ex.dt_path}')
    print(f'[gated] GT for scoring (never to model): {ex.img_root}/<set>/gts')

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

    if not getattr(opt.Model, 'use_dual_heads', False):
        print('[gated] WARNING: use_dual_heads is False in this config. '
              'Uncertainty will be degenerate and the gate falls back to '
              'hard replace at every tau.')

    # cache[testset] = list of (coarse_bool, gt_bool, dets, pred_ds, unc_ds)
    cache = {ts: [] for ts in opt.Test.Dataset.datasets}

    for testset in opt.Test.Dataset.datasets:
        img_path = osp.join(ex.img_root, testset, 'images')   # RGB INPUT
        gt_path = osp.join(ex.img_root, testset, 'gts')        # GT, scoring only
        mask_path = osp.join(ex.dt_path, testset)              # coarse preds
        ds = eval(opt.Test.Dataset.type)(
            img_root=img_path, mask_root=mask_path,
            transform_list=opt.Test.Dataset.transform_list)
        loader = data.DataLoader(ds, batch_size=1,
                                 num_workers=opt.Test.Dataloader.num_workers,
                                 pin_memory=opt.Test.Dataloader.pin_memory)
        for sample in loader:
            coarse = sample['gt'].squeeze(1)          # (1,H,W) coarse mask
            image = sample['image']                   # (1,3,H,W) RGB
            name = sample['name'][0]
            H, W = coarse.shape[-2], coarse.shape[-1]

            dets, img_patches, dt_patches = split(image, coarse)
            coarse_bool = (coarse.squeeze(0).cpu().numpy() > 0.5)
            gt_bool = _load_gt(gt_path, name, (H, W))

            if dets is None:
                # no boundary patches -> refined == coarse for all taus
                cache[testset].append((coarse_bool, gt_bool,
                                       np.zeros((0, 4)), np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32),
                                       np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.float32)))
                continue

            preds, uncs = [], []
            for i in range(0, len(img_patches), 8):
                p, u = tta_predict_unc(model, img_patches[i:i + 8], dt_patches[i:i + 8])
                # downsample to patch_size (mirrors merge()'s interpolation)
                p = F.interpolate(p, (PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
                u = F.interpolate(u, (PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
                preds.append(p.squeeze(1).cpu().numpy())
                uncs.append(u.squeeze(1).cpu().numpy())
            pred_ds = np.concatenate(preds, 0).astype(np.float32)
            unc_ds = np.concatenate(uncs, 0).astype(np.float32)
            dets_np = torch.cat(dets, 0).cpu().numpy() if isinstance(dets, list) else dets.cpu().numpy()
            cache[testset].append((coarse_bool, gt_bool, dets_np, pred_ds, unc_ds))
        print(f'[gated] cached {testset}: {len(cache[testset])} images')

    # ---- sweep ----
    print('\n' + '=' * 78)
    header = f'{"tau":>6} | ' + ' | '.join(f'{ts[:10]:>10}' for ts in opt.Test.Dataset.datasets) + ' | ' + f'{"MEAN":>7}'
    print(header)
    print('-' * len(header))
    for tau in taus:
        per_set_means = []
        for testset in opt.Test.Dataset.datasets:
            dices = []
            for (coarse_bool, gt_bool, dets_np, pred_ds, unc_ds) in cache[testset]:
                if dets_np.shape[0] == 0:
                    pred_bool = coarse_bool
                else:
                    pred_bool = merge_gated(coarse_bool, dets_np, pred_ds, unc_ds, tau)
                dices.append(dice_bin(pred_bool, gt_bool))
            per_set_means.append(float(np.mean(dices)))
        mean = float(np.mean(per_set_means))
        row = f'{tau:>6.2f} | ' + ' | '.join(f'{v:>10.4f}' for v in per_set_means) + ' | ' + f'{mean:>7.4f}'
        print(row)
    print('=' * 78)
    print('Anchors: tau=1.00 should match the current run; tau=0.00 should '
          'match the raw coarse base. If not, the scoring path is wrong.')
