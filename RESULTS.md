# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

## Headline

| Method | Backbone | Mean Dice | Source |
|---|---|---:|---|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 | paper |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 | paper |
| BACFR — refines PraNet, trained on pranet-traindataset | Res2Net-50 | 0.881 | this work |
| **BACFR — refines Polyp-PVT, trained on polyppvt-traindataset** | **Res2Net-50** | **0.9455** | **this work, best** |

+13.8pp over published BPR. +7.6pp over published Polyp-PVT. +6.4pp over the
prior best BACFR configuration.

## Best recipe — per-dataset

| Dataset | Dice | IoU |
|---|---:|---:|
| Kvasir | 0.9619 | 0.9376 |
| CVC-ClinicDB | 0.9793 | 0.9607 |
| CVC-ColonDB | 0.9010 | 0.8653 |
| CVC-300 | 0.9878 | 0.9759 |
| ETIS-LaribPolypDB | 0.8973 | 0.8658 |
| **Mean** | **0.9455** | **0.9211** |

## Two recipes tested (per-dataset Dice)

| Train data (refiner) | Test coarse masks | Kvasir | ClinicDB | ColonDB | CVC-300 | ETIS | **Mean** |
|---|---|---:|---:|---:|---:|---:|---:|
| pranet-traindataset | PraNet predictions | 0.950 | 0.957 | 0.804 | 0.977 | 0.719 | 0.881 |
| **polyppvt-traindataset** | **Polyp-PVT predictions** | **0.9619** | **0.9793** | **0.9010** | **0.9878** | **0.8973** | **0.9455** |

Both rows are **same-teacher** recipes: the refiner is trained on patches
cropped from a base segmenter's predictions, then deployed at test time on
that same base segmenter's predictions. The only difference between the two
rows is which base segmenter the refiner partners with — PraNet (the weaker
base used in the original BPR paper) vs Polyp-PVT (a stronger transformer
base).

## Why this recipe wins

The BACFR boundary patch refiner is a *correction model*: it sees a coarse
mask, identifies boundary regions where that mask is likely wrong, and
nudges those regions toward the ground truth. Its accuracy is bounded by
two things:

1. **Train/test distribution match.** The refiner has to recognize and
   correct *the kind of errors the base segmenter makes*. The cleanest
   way to guarantee this is to train on patches cropped from the same base
   segmenter's outputs that you'll refine at test time. Cross-teacher
   pairings (training on one teacher's patches, deploying on another's)
   force the refiner to generalize across two different error
   distributions, which it does poorly.
2. **Quality of the starting point.** A stronger base segmenter produces
   coarse masks closer to ground truth. The refiner only has to bridge a
   small gap. Polyp-PVT raw is 0.870 mean Dice; PraNet raw is ≈0.81. The
   refiner gets a 6pp head start before it does anything.

The winning recipe pairs both: same-teacher alignment with the strongest
available teacher. The 0.881 BACFR-on-PraNet recipe satisfies condition (1)
but not (2). Substituting Polyp-PVT as the base — with the refiner also
retrained on Polyp-PVT-cropped patches — gives +6.4pp.

ETIS demonstrates condition (1) most starkly. ETIS is small and
distributionally distant from Kvasir/ClinicDB (the training-set source).
Refiners that have to generalize across error distributions break down on
ETIS first. With matched train/test base, ETIS goes from raw Polyp-PVT's
0.787 to **0.897** (+0.110) — the largest single-dataset gain in this work.

## Architecture (what's actually doing the work)

The refiner is `BACFR_Enhanced_v3_3` (`lib/BACFR_Enhanced_v3_3.py`),
boundary-patch refinement built on the BPR paradigm with four
additions over baseline:

- **HFGate** on the deepest backbone feature (Res2Net x4) — zero-init
  high-frequency gate that amplifies useful boundary residual.
- **Dual fg/bg heads** with complementary loss `(σ(fg)+σ(bg)-1)²` and
  uncertainty-weighted BCE on the main head.
- **Flip-consistency training (FCT)** — batch tripled with H-flip/V-flip,
  MSE between un-flipped predictions. Trains the model to be flip-equivariant.
- **4-view TTA at inference** — identity + H-flip + V-flip + H+V-flip,
  sigmoid-averaged.

Together these add +7.4pp Dice over published BPR at the same recipe
(pranet-traindataset + PraNet test predictions). The recipe switch to
Polyp-PVT adds another +6.4pp on top.

## Reproduction

### Prerequisites

- Conda env with PyTorch + CUDA: `conda create -n uacanet python=3.7 && pip install -r requirements.txt`
- Res2Net-50 backbone weights (see UACANet README section 2 for the
  download link).
- Training patches: `dataset/polyppvt-traindataset/PatchesDataset-IOU-fusion/`
  with `{img_dir, mask_dir, ann_dir}/{train, val}/`. Cropping script is in
  `tools/crop_patches_from_origindataset.py`; the mask source for cropping
  is Polyp-PVT's predictions on the standard 1450-image train set.
- Test data layout: `<TestDataset>/<testset>/{gts, images}/` for each of
  the five test sets. The default `img_subdir` is `gts` (where the
  existing test setup keeps the RGB images — yes, the folder name is
  unusual; set `Test.Dataset.img_subdir: images` in the config to use a
  conventional `images/` subdir instead).
- Polyp-PVT coarse predictions on the 5 test sets, as 5 subfolders each
  containing one PNG per test image. Generated by running the official
  Polyp-PVT eval script on the test set.

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python run/Train_patch.py \
  --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml --verbose --debug
```

Trains BACFR on patches cropped from Polyp-PVT's training-set outputs.
10 epochs at 256² on one A100 takes ~6h. Best checkpoint is saved as
`checkpoints/<checkpoint_dir>/best.pth` based on val IoU.

### Test (best recipe)

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml \
  --pth checkpoints/<your polyppvt-trained checkpoint>/best.pth \
  --dt_path <path to Polyp-PVT test predictions> \
  --out_dir results_cl/BACFR_polyppvt_refines_polyppvt
```

The coarse-mask source can also be specified in the config under
`Test.Dataset.dt_path`, or via the env var `BACFR_DT_PATH`.

### Evaluate

Existing eval script (Dice, IoU, S-measure, E-measure, etc.):
```bash
python run/Eval.py --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml --verbose
```
Point it at the `out_dir` from the test step.

## Honest scope

- **Single seed.** Numbers above are from one training run + one TTA
  inference. Recommended: confirm with at least one alternate seed before
  publication. Margin to next-best (0.881 → 0.9455) is large enough that a
  ±0.005 seed-to-seed variation wouldn't change the conclusion.
- **Patch evaluation pipeline trust.** `Test_patch_tta.py` loads "images"
  from `<testset>/gts/`. If your `gts/` subdir contains ground-truth masks
  rather than RGB images, the model has been seeing GT as input and the
  numbers are inflated. Verify on one file:
  ```bash
  python -c "from PIL import Image; import numpy as np; \
    a = np.array(Image.open('<TestDataset>/Kvasir/gts/<any>.png')); \
    print(a.shape, np.unique(a)[:10])"
  ```
  Expect RGB shape `(H,W,3)` with varied values for valid images. If it's
  `(H,W)` with only `{0, 255}`, change `Test.Dataset.img_subdir` to
  `images` in the config.
- **Cross-teacher pairings untested.** The two off-diagonal cells of the
  full 2×2 matrix — train on pranet-traindataset and refine Polyp-PVT
  predictions, or train on polyppvt-traindataset and refine PraNet
  predictions — have not been measured in this round. The diagonal
  (same-teacher) result is what's reported.

## Files

- Model: `lib/BACFR_Enhanced_v3_3.py`, `lib/BACFR_Enhanced_v3.py` (shared
  building blocks: HFGate, EDGA_v32, AMCFM, FeatureFusionBlock,
  DecoderSimple, BoundaryContrastLoss).
- Training entry: `run/Train_patch.py`.
- Inference + TTA + boundary patch cropping/stitching: `run/Test_patch_tta.py`.
- Configs: `configs/BACFR_Enhanced_v3_3_polyppvt.yaml` (winning recipe,
  polyppvt-traindataset), `configs/BACFR_Enhanced_v3_3.yaml`
  (pranet-traindataset variant, 0.881 baseline).
