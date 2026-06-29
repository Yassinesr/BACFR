"""Oracle upper bounds for the patch-refinement pipeline.

Before iterating further on the polyppvt->Polyp-PVT recipe (or training
the mixed-data model), measure the *maximum possible* mean Dice given:
  * the current model's refined predictions
  * the current coarse base predictions
  * the current patch-selection heuristic in split()

Three oracle bounds (all use GT to make decisions; they're DIAGNOSTICS,
not deployable strategies):

  raw_coarse           : dice(coarse, gt)  -- lower anchor (no refinement)
  fully_refined        : dice(merge_hard(coarse, refined), gt)  -- current
  oracle_pixel         : at every touched pixel, pick {coarse, refined}
                         to match gt. Untouched pixels keep coarse.
                         STRICT UPPER BOUND for any pixel-level strategy.
  oracle_image         : per image, pick max(dice(coarse), dice(refined)).
                         Bound for an image-level refine/skip policy.

If oracle_pixel >> fully_refined, our merge throws away gains the
refiner already has -- gating/blending strategies have room.
If oracle_pixel ~= fully_refined, the refiner doesn't have the right
answer often enough -- gating can't help; need a better refiner or to
accept the ceiling.

GROUND TRUTH IS USED ONLY FOR SCORING AND ORACLE DECISIONS. Never fed
to the model. Model input is strictly (RGB image patch, coarse-mask
patch), same as the standard inference path.
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
# TTA over predictions only (no dual-head dependency)
# ==============================================================
@torch.no_grad()
def tta_pred(model, img_patches, dt_patches):
    views = [None, [3], [2], [2, 3]]
    p_sum = None
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
        p_sum = p if p_sum is None else p_sum + p
    return p_sum / 4.0


# ==============================================================
# Per-image: build (coarse_bool, refined_bool, touched_mask)
# refined_bool is the merged refined mask at touched pixels, else 0
# ==============================================================
def assemble_refined(coarse_bool, dets, pred_ds, patch_size=PATCH_SIZE):
    H, W = coarse_bool.shape
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = [int(v) for v in dets[i][:4]]
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            continue
        pr = pred_ds[i]
        if (h, w) != (patch_size, patch_size):
            pr = np.array(Image.fromarray(pr).resize((w, h), Image.BILINEAR))
        acc[y1:y2, x1:x2] += pr
        cnt[y1:y2, x1:x2] += 1
    touched = cnt > 0
    refined = np.zeros_like(coarse_bool)
    if touched.any():
        avg = acc[touched] / cnt[touched]
        refined[touched] = avg > 0.5
    return refined, touched


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


# ==============================================================
# Main
# ==============================================================
if __name__ == '__main__':
    args = parse_args()
    config = args.config if os.path.isfile(args.config) else 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[oracle] using config: {config}')
    opt = load_config(config)

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pth', type=str, default=None)
    extra.add_argument('--dt_path', type=str, required=True)
    extra.add_argument('--img_root', type=str,
                       default='/home/yassine/projects/UACANet-main/dataset/TestDataset')
    ex, _ = extra.parse_known_args()

    pth = ex.pth or osp.join(opt.Test.Checkpoint.checkpoint_dir, 'best.pth')
    print(f'[oracle] checkpoint: {pth}')
    print(f'[oracle] coarse masks: {ex.dt_path}')
    print(f'[oracle] GT (scoring + oracle decisions only, not to model): '
          f'{ex.img_root}/<set>/gts')

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

    # accumulator[testset][bound_name] = list of per-image Dice
    bounds = ['raw_coarse', 'fully_refined', 'oracle_image', 'oracle_pixel',
              'oracle_geometric']
    results = {ts: {b: [] for b in bounds} for ts in opt.Test.Dataset.datasets}

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
                refined_bool = np.zeros_like(coarse_bool)
                touched = np.zeros_like(coarse_bool)
            else:
                preds = []
                for i in range(0, len(img_patches), 8):
                    p = tta_pred(model, img_patches[i:i + 8], dt_patches[i:i + 8])
                    p = F.interpolate(p, (PATCH_SIZE, PATCH_SIZE),
                                      mode='bilinear', align_corners=False)
                    preds.append(p.squeeze(1).cpu().numpy())
                pred_ds = np.concatenate(preds, 0).astype(np.float32)
                dets_np = (torch.cat(dets, 0).cpu().numpy()
                           if isinstance(dets, list) else dets.cpu().numpy())
                refined_bool, touched = assemble_refined(coarse_bool, dets_np, pred_ds)

            # raw_coarse: just the coarse mask, no refinement
            d_raw = dice_bin(coarse_bool, gt_bool)

            # fully_refined: coarse with refined overwriting touched pixels
            fr_out = coarse_bool.copy()
            if touched.any():
                fr_out[touched] = refined_bool[touched]
            d_fr = dice_bin(fr_out, gt_bool)

            # oracle_pixel: at touched pixels, peek GT and pick the value
            # (coarse or refined) that matches. Untouched -> coarse.
            op_out = coarse_bool.copy()
            if touched.any():
                # at each touched pixel, oracle output = gt if either
                # coarse or refined matches gt; else coarse (still wrong)
                either_correct = ((coarse_bool == gt_bool) | (refined_bool == gt_bool))
                pick = touched & either_correct
                op_out[pick] = gt_bool[pick]
            d_op = dice_bin(op_out, gt_bool)

            # oracle_image: per image, pick max of raw_coarse vs fully_refined
            d_oi = max(d_raw, d_fr)

            # oracle_geometric (Bound C): ASSUME a perfect refiner that always
            # outputs GT at every touched pixel. This removes the dependency on
            # which refiner we trained -- it bounds the patch-refinement
            # PARADIGM itself on this base segmenter. The remaining error is
            # purely structural: coarse-mask pixels that disagree with GT but
            # are never touched by any boundary patch (missed regions). No
            # refiner / gate / ensemble / training trick can exceed this.
            geo_out = coarse_bool.copy()
            if touched.any():
                geo_out[touched] = gt_bool[touched]
            d_geo = dice_bin(geo_out, gt_bool)

            results[testset]['raw_coarse'].append(d_raw)
            results[testset]['fully_refined'].append(d_fr)
            results[testset]['oracle_image'].append(d_oi)
            results[testset]['oracle_pixel'].append(d_op)
            results[testset]['oracle_geometric'].append(d_geo)

        print(f'[oracle] processed {testset}: {len(results[testset]["raw_coarse"])} images')

    # ---- table ----
    print('\n' + '=' * 100)
    sets = opt.Test.Dataset.datasets
    col = '{:>14}'
    header = '{:>17} | '.format('bound') + ' | '.join(col.format(s[:13]) for s in sets) + ' | ' + col.format('MEAN')
    print(header)
    print('-' * len(header))
    summary = {}
    for b in bounds:
        per_set = [float(np.mean(results[ts][b])) for ts in sets]
        mean = float(np.mean(per_set))
        summary[b] = mean
        row = '{:>17} | '.format(b) + ' | '.join(col.format(f'{v:.4f}') for v in per_set) + ' | ' + col.format(f'{mean:.4f}')
        print(row)
    print('=' * 100)

    # ---- interpretation ----
    headroom_image = summary['oracle_image'] - summary['fully_refined']
    headroom_pixel = summary['oracle_pixel'] - summary['fully_refined']
    headroom_geo = summary['oracle_geometric'] - summary['fully_refined']
    print()
    print(f'Current (fully_refined):              {summary["fully_refined"]:.4f}')
    print(f'Headroom to oracle_image:             +{headroom_image:.4f}  '
          f'(perfect image-level refine/skip policy)')
    print(f'Headroom to oracle_pixel:             +{headroom_pixel:.4f}  '
          f'(perfect gate over {{coarse, THIS refiner}})')
    print(f'Headroom to oracle_geometric (ABS):   +{headroom_geo:.4f}  '
          f'(perfect refiner -> GT at every touched pixel)')
    print()
    print(f'ABSOLUTE PARADIGM CEILING (geometric): {summary["oracle_geometric"]:.4f}')
    print(f'  This is the max Dice the patch-refinement paradigm can reach on')
    print(f'  THIS base segmenter, with ANY refiner. The gap from 1.0 is error')
    print(f'  that lives outside the boundary patches (missed regions) -- a')
    print(f'  property of the base segmenter + split() heuristic, not the model.')
    print()
    # How much of the absolute ceiling does a better REFINER (not just gating)
    # still have to give? gap between the geometric ceiling and what a perfect
    # gate on the current refiner would reach.
    refiner_room = summary['oracle_geometric'] - summary['oracle_pixel']
    print(f'Refiner-improvement room:             +{refiner_room:.4f}  '
          f'(geometric ceiling - current-refiner oracle)')
    print(f'  If large, a BETTER REFINER (e.g. error-focus, retrain) can still')
    print(f'  climb. If ~0, the current refiner already has every reachable')
    print(f'  pixel and only gating slack remains.')
    print()
    if headroom_pixel < 0.005:
        print('VERDICT: ~no headroom. The refiner makes ~the same calls as coarse.')
        print('         Inference-time tweaks cannot meaningfully improve this recipe.')
        print('         Better refinement requires a better refiner or paradigm change.')
    elif headroom_pixel < 0.015:
        print('VERDICT: modest headroom. A smart gating policy might recover')
        print('         half of this -- expect +0.003-0.008 from gating efforts.')
        print('         Mixed-data training is probably a similar-magnitude gamble.')
    else:
        print('VERDICT: meaningful headroom. The refiner has correct answers')
        print('         our merge throws away. Smarter gating, mixed training,')
        print('         or both could each push toward this ceiling.')
