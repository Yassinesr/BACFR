"""Ablation baselines — refiner contribution and the value of TTA.

Reports three rows scored by ONE internal harness so the ablation column
is internally consistent:

  raw_coarse        the base segmenter's mask, no refinement
  refine_no_tta     single forward (identity only) + standard hard merge
  refine_tta        4-view flip TTA + standard hard merge  (== the
                    deployed "normal TTA, no gating" number)

The gap (refine_tta - refine_no_tta) is the value of TTA on this recipe.
The gap (refine_no_tta - raw_coarse) is the value of the refiner before
any test-time augmentation.

Standard hard merge = overwrite the coarse mask with refined>0.5 at every
touched pixel (no gating). Matches run/Test_patch_tta.py's merge().

GROUND TRUTH IS USED ONLY FOR SCORING. Never fed to the model.
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


@torch.no_grad()
def predict(model, img_patches, dt_patches, tta):
    """Return mean sigmoid prediction. tta=False -> single identity forward;
    tta=True -> 4-view flip average (un-flipped)."""
    views = [None, [3], [2], [2, 3]] if tta else [None]
    p_sum = None
    for fd in views:
        if fd is None:
            s = {'image': img_patches, 'mask': dt_patches}
        else:
            s = {'image': torch.flip(img_patches, dims=fd),
                 'mask':  torch.flip(dt_patches, dims=fd)}
        p = torch.sigmoid(model(to_cuda(s))['pred'])
        if fd is not None:
            p = torch.flip(p, dims=fd)
        p_sum = p if p_sum is None else p_sum + p
    return p_sum / len(views)


def assemble_hard(coarse_bool, dets, pred_ds, patch_size=PATCH_SIZE):
    """Standard hard merge: overwrite coarse with refined>0.5 at touched."""
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
    out = coarse_bool.copy()
    m = cnt > 0
    if m.any():
        out[m] = (acc[m] / cnt[m]) > 0.5
    return out


def dice_bin(pred_bool, gt_bool):
    na = np.logical_and(pred_bool, gt_bool).sum()
    if na == 0:
        return 0.0
    return float(2 * na / (gt_bool.sum() + pred_bool.sum()))


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
    print(f'[ablation] using config: {config}')
    opt = load_config(config)

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pth', type=str, default=None)
    extra.add_argument('--dt_path', type=str, required=True)
    extra.add_argument('--img_root', type=str,
                       default='/home/yassine/projects/UACANet-main/dataset/TestDataset')
    ex, _ = extra.parse_known_args()

    pth = ex.pth or osp.join(opt.Test.Checkpoint.checkpoint_dir, 'best.pth')
    print(f'[ablation] checkpoint: {pth}')
    print(f'[ablation] coarse base: {ex.dt_path}')

    model = eval(opt.Model.name)(
        channels=opt.Model.channels, output_stride=opt.Model.output_stride,
        pretrained=False,
        use_mccpb=getattr(opt.Model, 'use_mccpb', False),
        use_dual_heads=getattr(opt.Model, 'use_dual_heads', False),
        use_boundary_contrast=getattr(opt.Model, 'use_boundary_contrast', False),
        use_hf_gate=getattr(opt.Model, 'use_hf_gate', False),
        use_flip_consistency=getattr(opt.Model, 'use_flip_consistency', False),
        edge_dist_mode=getattr(opt.Model, 'edge_dist_mode', 'cdist'))
    ck = torch.load(pth, map_location='cuda')
    st = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    st = st.get('state_dict', st) if isinstance(st, dict) and 'state_dict' in st else st
    model.load_state_dict(st, strict=True)
    model.cuda().eval()

    rows = ['raw_coarse', 'refine_no_tta', 'refine_tta']
    res = {ts: {r: [] for r in rows} for ts in opt.Test.Dataset.datasets}

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

            res[testset]['raw_coarse'].append(dice_bin(coarse_bool, gt_bool))

            dets, img_patches, dt_patches = split(image, coarse)
            if dets is None:
                res[testset]['refine_no_tta'].append(dice_bin(coarse_bool, gt_bool))
                res[testset]['refine_tta'].append(dice_bin(coarse_bool, gt_bool))
                continue
            dets_np = (torch.cat(dets, 0).cpu().numpy()
                       if isinstance(dets, list) else dets.cpu().numpy())

            for mode, tta in (('refine_no_tta', False), ('refine_tta', True)):
                preds = []
                for i in range(0, len(img_patches), 8):
                    p = predict(model, img_patches[i:i + 8], dt_patches[i:i + 8], tta)
                    p = F.interpolate(p, (PATCH_SIZE, PATCH_SIZE),
                                      mode='bilinear', align_corners=False)
                    preds.append(p.squeeze(1).cpu().numpy())
                pred_ds = np.concatenate(preds, 0).astype(np.float32)
                out = assemble_hard(coarse_bool, dets_np, pred_ds)
                res[testset][mode].append(dice_bin(out, gt_bool))
        print(f'[ablation] processed {testset}: {len(res[testset]["raw_coarse"])} imgs')

    sets = opt.Test.Dataset.datasets
    print('\n' + '=' * 100)
    col = '{:>14}'
    header = '{:>16} | '.format('row') + ' | '.join(col.format(s[:13]) for s in sets) + ' | ' + col.format('MEAN')
    print(header)
    print('-' * len(header))
    summ = {}
    for r in rows:
        per = [float(np.mean(res[ts][r])) for ts in sets]
        summ[r] = float(np.mean(per))
        line = '{:>16} | '.format(r) + ' | '.join(col.format(f'{v:.4f}') for v in per) + ' | ' + col.format(f'{summ[r]:.4f}')
        print(line)
    print('=' * 100)
    print()
    print(f'Refiner value (no TTA) : {summ["refine_no_tta"] - summ["raw_coarse"]:+.4f}  '
          f'(refine_no_tta - raw_coarse)')
    print(f'TTA value              : {summ["refine_tta"] - summ["refine_no_tta"]:+.4f}  '
          f'(refine_tta - refine_no_tta)')
    print(f'Total refiner+TTA value: {summ["refine_tta"] - summ["raw_coarse"]:+.4f}')
