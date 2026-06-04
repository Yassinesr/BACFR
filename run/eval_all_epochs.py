"""
Evaluate every saved epoch checkpoint of a training run.

Runs the full patch-refinement pipeline for each epoch on all 5 test datasets,
saves refined masks, then computes Dice/IoU. Prints a per-epoch summary table
and a "best per dataset" recommendation at the end.

Usage (from repo root):
    python eval_all_epochs.py --ckpt_dir checkpoints/BACFR_v32_hfgate_dual --config configs/BACFR_Enhanced_v3.yaml
    python eval_all_epochs.py --ckpt_dir checkpoints/BACFR_v32_hfgate          # uses default config
    python eval_all_epochs.py --ckpt_dir checkpoints/BACFR_v32_hfgate_dual --epochs 5 6 7 8 9 10
"""

import os
import os.path as osp
import sys
import argparse
import shutil

import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2

from torchvision.ops import nms, roi_align

# Force repo root onto sys.path BEFORE any project imports
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from utils.utils import load_config, to_cuda
from utils.dataloader import *
from lib import *


# ==============================================================
# Patch refinement helpers (copied from Test_patch.py)
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


# ==============================================================
# Refinement pass over all test datasets
# ==============================================================
def run_refinement(opt, model, out_dir, dt_path):
    os.makedirs(out_dir, exist_ok=True)

    for testset in opt.Test.Dataset.datasets:
        save_dir = os.path.join(out_dir, testset)
        os.makedirs(save_dir, exist_ok=True)

        root = "/home/yassine/projects/UACANet-main/dataset/TestDataset"
        # If you're on Windows, change this to the Windows path:
        # root = r"C:\Users\hp\Desktop\UACANet-main\dataset\TestDataset"
        img_path = os.path.join(root, testset, 'images')
        mask_path = os.path.join(dt_path, testset)

        test_dataset = eval(opt.Test.Dataset.type)(
            img_root=img_path, mask_root=mask_path,
            transform_list=opt.Test.Dataset.transform_list
        )

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=opt.Test.Dataloader.num_workers,
            pin_memory=opt.Test.Dataloader.pin_memory
        )

        for sample in test_loader:
            mask = sample['gt'].squeeze(1)
            image = sample['image']
            s = {}

            dets, img_patches, dt_patches = split(image, mask)
            if dets is None:
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

            refinemasks_final = torch.tensor(refinemasks_final).cuda()
            refineds = merge(mask.cuda(), dets, refinemasks_final)
            for i in range(len(refineds)):
                cv2.imwrite(
                    osp.join(save_dir, sample['name'][0]),
                    refineds[i].cpu().numpy().astype(np.uint8) * 255
                )


# ==============================================================
# Dice/IoU computation
# ==============================================================
def compute_metrics(pred_dir, gt_dir):
    """Compute mean Dice and mean IoU over all images in a dataset folder."""
    dices, ious = [], []
    for fname in sorted(os.listdir(gt_dir)):
        gt_path = osp.join(gt_dir, fname)
        pred_path = osp.join(pred_dir, fname)
        if not osp.exists(pred_path):
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue

        # Resize pred to match gt if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        gt_b = (gt > 127).astype(np.float32)
        pred_b = (pred > 127).astype(np.float32)

        inter = (gt_b * pred_b).sum()
        union = gt_b.sum() + pred_b.sum()
        dice = (2 * inter) / (union + 1e-8) if union > 0 else 1.0

        u = ((gt_b + pred_b) > 0).sum()
        iou = inter / (u + 1e-8) if u > 0 else 1.0

        dices.append(dice)
        ious.append(iou)

    return float(np.mean(dices)) if dices else 0.0, \
           float(np.mean(ious)) if ious else 0.0


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory containing epoch_*.pth files')
    parser.add_argument('--config', type=str, default='configs/BACFR_Enhanced_v3.yaml',
                        help='YAML config path')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='Specific epochs to evaluate (default: all found)')
    parser.add_argument('--dt_path', type=str,
                        default='/home/yassine/projects/UACANet-main/results_cl/paper_results/PraNet-results/PraNet',
                        help='Path to PraNet coarse masks (the input to refine)')
    parser.add_argument('--gt_root', type=str,
                        default='/home/yassine/projects/UACANet-main/dataset/TestDataset',
                        help='Root of test dataset (contains masks per dataset)')
    parser.add_argument('--results_root', type=str, default='results_cl',
                        help='Where to save refined masks per epoch')
    parser.add_argument('--keep_results', action='store_true',
                        help='Keep refined mask folders after evaluation (default: delete)')
    args = parser.parse_args()

    # Load config
    opt = load_config(args.config)

    # Build model template once
    model = eval(opt.Model.name)(
        channels=opt.Model.channels,
        output_stride=opt.Model.output_stride,
        pretrained=False,  # we're loading checkpoints, no need to download pretrained
        use_mccpb=getattr(opt.Model, 'use_mccpb', False),
        use_dual_heads=getattr(opt.Model, 'use_dual_heads', False),
        use_boundary_contrast=getattr(opt.Model, 'use_boundary_contrast', False),
        use_hf_gate=getattr(opt.Model, 'use_hf_gate', False),
        edge_dist_mode=getattr(opt.Model, 'edge_dist_mode', 'cdist'),
    )
    model = model.cuda()

    # Find available epoch checkpoints
    if args.epochs is None:
        epochs = []
        for f in os.listdir(args.ckpt_dir):
            if f.startswith('epoch_') and f.endswith('.pth'):
                try:
                    epochs.append(int(f.replace('epoch_', '').replace('.pth', '')))
                except ValueError:
                    pass
        epochs = sorted(epochs)
    else:
        epochs = sorted(args.epochs)

    if not epochs:
        print(f"ERROR: no epoch_*.pth files found in {args.ckpt_dir}")
        return

    print(f"\nEvaluating epochs: {epochs}")
    print(f"Checkpoint dir:    {args.ckpt_dir}")
    print(f"Config:            {args.config}\n")

    # Results: {epoch: {dataset: (dice, iou)}}
    results = {}
    run_tag = osp.basename(args.ckpt_dir.rstrip('/\\'))

    for epoch in epochs:
        pth = osp.join(args.ckpt_dir, f'epoch_{epoch}.pth')
        if not osp.exists(pth):
            print(f"  Skipping epoch {epoch}: {pth} not found")
            continue

        print(f"\n{'='*70}\n  EPOCH {epoch}\n{'='*70}")

        # Load checkpoint
        state = torch.load(pth, map_location='cuda')
        msg = model.load_state_dict(state, strict=False)
        if msg.missing_keys:
            print(f"  Warning: missing keys: {len(msg.missing_keys)} (likely OK)")
        if msg.unexpected_keys:
            print(f"  Warning: unexpected keys: {len(msg.unexpected_keys)}")

        model.eval()

        # Run refinement on all 5 datasets
        out_dir = osp.join(args.results_root, f'{run_tag}_ep{epoch}')
        with torch.no_grad():
            run_refinement(opt, model, out_dir, args.dt_path)

        # Compute metrics per dataset
        epoch_results = {}
        for testset in opt.Test.Dataset.datasets:
            pred_dir = osp.join(out_dir, testset)
            gt_dir = osp.join(args.gt_root, testset, 'gts')
            if not osp.isdir(gt_dir):
                print(f"  WARNING: no GT folder found for {testset}")
                continue
            d, i = compute_metrics(pred_dir, gt_dir)
            epoch_results[testset] = (d, i)
            print(f"    {testset:25s}  Dice {d:.4f}  IoU {i:.4f}")

        results[epoch] = epoch_results

        # Optional: delete refined masks folder to save disk
        if not args.keep_results:
            try:
                shutil.rmtree(out_dir)
            except Exception:
                pass

    # ==========================================================
    # Print summary table
    # ==========================================================
    if not results:
        print("No epochs were evaluated.")
        return

    datasets = list(opt.Test.Dataset.datasets)

    print(f"\n\n{'='*100}")
    print(f"  PER-EPOCH DICE SUMMARY  ({run_tag})")
    print(f"{'='*100}")

    header = f"{'Epoch':>6}  " + "  ".join(f"{d:>20s}" for d in datasets)
    print(header)
    print("-" * len(header))

    for epoch in sorted(results.keys()):
        row = f"{epoch:>6d}  "
        for d in datasets:
            if d in results[epoch]:
                row += f"{results[epoch][d][0]:>20.4f}  "
            else:
                row += f"{'--':>20s}  "
        print(row)

    # ==========================================================
    # Best per dataset
    # ==========================================================
    print(f"\n{'='*100}")
    print(f"  BEST PER DATASET")
    print(f"{'='*100}")
    print(f"{'Dataset':<25s}  {'Best Dice':>10s}  {'Best IoU':>10s}  {'@ Epoch':>10s}")
    print("-" * 65)

    best_table = {}
    for d in datasets:
        best_epoch = None
        best_dice = -1.0
        best_iou = -1.0
        for epoch, er in results.items():
            if d in er and er[d][0] > best_dice:
                best_dice = er[d][0]
                best_iou = er[d][1]
                best_epoch = epoch
        if best_epoch is not None:
            best_table[d] = (best_dice, best_iou, best_epoch)
            print(f"{d:<25s}  {best_dice:>10.4f}  {best_iou:>10.4f}  {best_epoch:>10d}")

    # ==========================================================
    # Comparison vs PraNet and BACFR
    # ==========================================================
    PRANET = {
        'Kvasir':            0.899,
        'CVC-ClinicDB':      0.905,
        'CVC-ColonDB':       0.715,
        'CVC-300':           0.877,
        'ETIS-LaribPolypDB': 0.636,
    }
    BACFR = {
        'Kvasir':            0.899,
        'CVC-ClinicDB':      0.916,
        'CVC-ColonDB':       0.725,
        'CVC-300':           0.877,
        'ETIS-LaribPolypDB': 0.648,
    }

    print(f"\n{'='*100}")
    print(f"  GAP ANALYSIS (best Dice across all evaluated epochs)")
    print(f"{'='*100}")
    print(f"{'Dataset':<25s}  {'Best':>8s}  {'PraNet':>8s}  {'vs PraNet':>10s}  {'BACFR':>8s}  {'vs BACFR':>10s}")
    print("-" * 90)

    win_count = 0
    plus_one_count = 0
    for d in datasets:
        if d not in best_table:
            continue
        best_d = best_table[d][0]
        pn = PRANET.get(d, None)
        bf = BACFR.get(d, None)
        pn_str = f"{pn:.3f}" if pn else "--"
        bf_str = f"{bf:.3f}" if bf else "--"

        if pn is not None:
            gap_pn = (best_d - pn) * 100
            pn_gap_str = f"{gap_pn:+.2f}%"
            if gap_pn > 0:
                win_count += 1
            if gap_pn >= 1.0:
                plus_one_count += 1
        else:
            pn_gap_str = "--"

        if bf is not None:
            gap_bf = (best_d - bf) * 100
            bf_gap_str = f"{gap_bf:+.2f}%"
        else:
            bf_gap_str = "--"

        print(f"{d:<25s}  {best_d:>8.4f}  {pn_str:>8s}  {pn_gap_str:>10s}  {bf_str:>8s}  {bf_gap_str:>10s}")

    print(f"\n  Datasets beating PraNet:       {win_count}/5")
    print(f"  Datasets with +1% over PraNet: {plus_one_count}/5")
    print()


if __name__ == '__main__':
    main()
