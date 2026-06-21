"""Multi-refiner oracle — GO/NO-GO diagnostic for the ceiling chase.

The single-refiner oracle (Test_patch_tta_oracle.py) showed the
polyppvt-trained refiner's pixel-wise ceiling on the Polyp-PVT base is
0.8854 -- a +0.013 cap that requires a perfect, unattainable gate.

This script answers two questions before we build ensemble/gating
machinery:

  1. Does each refiner (polyppvt, mixed, pranet) have a DIFFERENT
     pixel-wise oracle on the Polyp-PVT base? (re-oracle all of them)
  2. Does a perfect selector over MULTIPLE refiners' outputs raise the
     ceiling? combined_oracle picks, per touched pixel, the value from
     {coarse, refined_1, refined_2, ...} that matches GT.

Reading the result:
  * combined_oracle >> best single oracle  -> refiners are complementary;
    an ensemble + good gate has a real target. Build it.
  * combined_oracle ~= best single oracle  -> refiners agree; combining
    them adds nothing. We are at the structural cap of the patch
    paradigm. Stop chasing; write the honest paper.

The gap (1 - combined_oracle) is error that lives OUTSIDE the refinable
boundary patches (missed regions, both-wrong pixels) -- unreachable by
any refiner-selection strategy. That is the true ceiling.

GT used ONLY for scoring + oracle decisions. Never fed to any model.
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
        refined[touched] = (acc[touched] / cnt[touched]) > 0.5
    return refined, touched


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


def build_model(opt, pth):
    m = eval(opt.Model.name)(
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
    m.load_state_dict(st, strict=True)
    return m.cuda().eval()


if __name__ == '__main__':
    args = parse_args()
    config = args.config if os.path.isfile(args.config) else 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[multi-oracle] using config (architecture): {config}')
    opt = load_config(config)

    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--pths', type=str, required=True,
                       help='comma-separated checkpoint paths (one per refiner)')
    extra.add_argument('--names', type=str, default=None,
                       help='comma-separated short names, same order as --pths')
    extra.add_argument('--dt_path', type=str, required=True)
    extra.add_argument('--img_root', type=str,
                       default='/home/yassine/projects/UACANet-main/dataset/TestDataset')
    ex, _ = extra.parse_known_args()

    pths = [p.strip() for p in ex.pths.split(',')]
    names = ([n.strip() for n in ex.names.split(',')] if ex.names
             else [f'refiner{i}' for i in range(len(pths))])
    assert len(names) == len(pths)
    print(f'[multi-oracle] refiners: {list(zip(names, pths))}')
    print(f'[multi-oracle] coarse base: {ex.dt_path}')

    models = [build_model(opt, p) for p in pths]

    sets = opt.Test.Dataset.datasets
    # per-set lists of dice values
    acc = {ts: {'raw': [], 'combined_oracle': [],
                **{f'refined::{n}': [] for n in names},
                **{f'oracle::{n}': [] for n in names}} for ts in sets}

    for testset in sets:
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
            refineds = {}
            touched = np.zeros_like(coarse_bool)
            if dets is not None:
                dets_np = (torch.cat(dets, 0).cpu().numpy()
                           if isinstance(dets, list) else dets.cpu().numpy())
                for n, model in zip(names, models):
                    preds = []
                    for i in range(0, len(img_patches), 8):
                        p = tta_pred(model, img_patches[i:i + 8], dt_patches[i:i + 8])
                        p = F.interpolate(p, (PATCH_SIZE, PATCH_SIZE),
                                          mode='bilinear', align_corners=False)
                        preds.append(p.squeeze(1).cpu().numpy())
                    pred_ds = np.concatenate(preds, 0).astype(np.float32)
                    rb, tch = assemble_refined(coarse_bool, dets_np, pred_ds)
                    refineds[n] = rb
                    touched = tch
            else:
                for n in names:
                    refineds[n] = np.zeros_like(coarse_bool)

            # raw
            acc[testset]['raw'].append(dice_bin(coarse_bool, gt_bool))

            # per-refiner fully_refined and single oracle
            for n in names:
                fr = coarse_bool.copy()
                if touched.any():
                    fr[touched] = refineds[n][touched]
                acc[testset][f'refined::{n}'].append(dice_bin(fr, gt_bool))

                op = coarse_bool.copy()
                if touched.any():
                    either = (coarse_bool == gt_bool) | (refineds[n] == gt_bool)
                    pick = touched & either
                    op[pick] = gt_bool[pick]
                acc[testset][f'oracle::{n}'].append(dice_bin(op, gt_bool))

            # combined oracle over {coarse} U {all refiners}
            co = coarse_bool.copy()
            if touched.any():
                either = (coarse_bool == gt_bool)
                for n in names:
                    either = either | (refineds[n] == gt_bool)
                pick = touched & either
                co[pick] = gt_bool[pick]
            acc[testset]['combined_oracle'].append(dice_bin(co, gt_bool))

        print(f'[multi-oracle] processed {testset}: {len(acc[testset]["raw"])} imgs')

    def mean_over_sets(key):
        return float(np.mean([np.mean(acc[ts][key]) for ts in sets]))

    print('\n' + '=' * 60)
    print(f'{"configuration":>28} | {"mean Dice":>10}')
    print('-' * 60)
    print(f'{"raw coarse base":>28} | {mean_over_sets("raw"):>10.4f}')
    for n in names:
        print(f'{("refined: " + n):>28} | {mean_over_sets("refined::" + n):>10.4f}')
    print('-' * 60)
    for n in names:
        print(f'{("oracle (1 refiner): " + n):>28} | {mean_over_sets("oracle::" + n):>10.4f}')
    combined = mean_over_sets('combined_oracle')
    print(f'{"COMBINED ORACLE (all)":>28} | {combined:>10.4f}')
    print('=' * 60)

    best_single_oracle = max(mean_over_sets('oracle::' + n) for n in names)
    best_refined = max(mean_over_sets('refined::' + n) for n in names)
    print()
    print(f'Best deployable today (refined):     {best_refined:.4f}')
    print(f'Best single-refiner oracle:          {best_single_oracle:.4f}')
    print(f'Combined multi-refiner oracle:       {combined:.4f}')
    print(f'Combining lift over best single:     +{combined - best_single_oracle:.4f}')
    print()
    if combined - best_single_oracle < 0.004:
        print('VERDICT: refiners agree. Combining them does NOT raise the ceiling.')
        print('         We are at the structural cap of the patch paradigm --')
        print('         remaining error is outside refinable boundary patches.')
        print('         Stop chasing; write the honest paper.')
    else:
        print('VERDICT: refiners are complementary. An ensemble + good gate has')
        print('         a real target. Worth building ensemble inference + a')
        print('         learned/variance gate to capture part of this.')
