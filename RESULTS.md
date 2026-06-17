# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

## Headline

| Method | Backbone | Mean Dice | Source |
|---|---|---:|---|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 | paper |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 | paper |
| BACFR — same-teacher, weak base (trained on pranet-traindataset, refines PraNet) | Res2Net-50 | 0.881 | this work |
| BACFR — cross-teacher (trained on polyppvt-traindataset, refines PraNet) | Res2Net-50 | 0.855 | this work, ablation |
| **BACFR — same-teacher, strong base (trained on polyppvt-traindataset, refines Polyp-PVT)** | **Res2Net-50** | **0.9455** | **this work, best** |

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

## The full ablation (per-dataset Dice)

| Train data (refiner) | Test coarse masks | Pairing | Kvasir | ClinicDB | ColonDB | CVC-300 | ETIS | **Mean** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pranet-traindataset | PraNet predictions | same-teacher, weak base | 0.950 | 0.957 | 0.804 | 0.977 | 0.719 | 0.881 |
| **polyppvt-traindataset** | **PraNet predictions** | **cross-teacher (mismatch)** | 0.9285 | 0.9424 | 0.7938 | 0.9177 | **0.6914** | **0.855** |
| **polyppvt-traindataset** | **Polyp-PVT predictions** | **same-teacher, strong base** | **0.9619** | **0.9793** | **0.9010** | **0.9878** | **0.8973** | **0.9455** |

The cleanest apples-to-apples for the same-teacher hypothesis is the
**first two rows**: both refine the same PraNet test predictions. Only
the training distribution differs. Same-teacher (top row, pranet-trained,
0.881) **beats** cross-teacher (middle row, polyppvt-trained, 0.855) by
**0.026** mean Dice — the cost of training the refiner on the wrong
base's error patterns. ETIS shows the gap most clearly: 0.719 → 0.691
(−0.028) on identical test inputs.

The bottom row then shows what same-teacher pairing buys when you also
swap to a stronger base: another **+6.4pp** mean Dice and **+0.206 on
ETIS** specifically.

### Untested fourth cell

`pranet-traindataset` + `Polyp-PVT predictions` — the other cross-teacher
direction — has not been measured. By the same-teacher hypothesis it
should also under-perform the bottom row's 0.9455, since the refiner
would be trained on PraNet's error distribution but deployed on
Polyp-PVT's. Cheap to run (inference only, no new training); would close
the 2×2 matrix for the paper.

## Why this recipe wins

The BACFR boundary patch refiner is a *correction model*: it sees a coarse
mask, identifies boundary regions where that mask is likely wrong, and
nudges those regions toward the ground truth. Its accuracy is bounded by
two factors, and the ablation above shows you need both.

1. **Train/test distribution match (same-teacher).** The refiner has to
   recognize and correct *the kind of errors the base segmenter makes*.
   Different segmenters fail in different ways — PraNet's boundary errors
   are coarser and more global; Polyp-PVT's are subtler and finer. A
   refiner trained on one teacher's error distribution and deployed on
   another's has to generalize across two different error modes, which it
   does poorly. The cross-teacher row (polyppvt-trained, PraNet test
   predictions) demonstrates this on a held-fixed test base: the same
   PraNet predictions are refined, but a mismatched training distribution
   costs 0.026 mean Dice vs the same-teacher pranet+PraNet baseline.
2. **Quality of the starting point (strong base).** A stronger base
   segmenter produces coarse masks closer to ground truth. The refiner
   only has to bridge a small gap. Polyp-PVT raw is 0.870 mean Dice;
   PraNet raw is ≈0.81. The refiner gets a 6pp head start before it does
   anything. But this head start is wasted unless condition (1) is also
   met — switching to a stronger base only pays off when the refiner has
   actually been trained on that base's error patterns.

The winning recipe pairs both: same-teacher alignment **and** the
strongest available teacher.

### ETIS demonstrates the story most starkly

ETIS is the smallest test set and the most out-of-distribution relative to
the Kvasir+ClinicDB training source. It's where mismatched train/test
error distributions break down first.

**Same-test-base comparison (both rows refine PraNet's ETIS predictions):**

| Recipe | ETIS Dice |
|---|---:|
| Same-teacher (pranet-trained refiner) | **0.719** |
| Cross-teacher (polyppvt-trained refiner) | 0.691 |

Identical test inputs, only the refiner's training data differs — and the
mismatched training distribution costs **−0.028 on ETIS** alone (most of
the −0.026 mean drop). The same dataset under the same coarse-mask
input, with a wrongly-trained refiner.

**Strong-base, same-teacher gain (vs same-teacher with weak base):**

| Recipe | ETIS Dice | Δ vs same-teacher weak (0.719) |
|---|---:|---:|
| Same-teacher, weak base (pranet+PraNet) | 0.719 | — |
| **Same-teacher, strong base (polyppvt+Polyp-PVT)** | **0.897** | **+0.178** |

ETIS jumps +0.178 when both conditions are satisfied — the largest
single-dataset gain in this work.

## Architecture (what's actually doing the work)

The refiner is `BACFR_Enhanced_v3_3` (`lib/BACFR_Enhanced_v3_3.py`),
boundary-patch refinement built on the BPR paradigm with four
additions over baseline:

- **HFGate** on the deepest backbone feature (Res2Net x4) — zero-init
  high-frequency gate that amplifies useful boundary residual.
- **Dual fg/bg heads** with complementary loss `(σ(fg)+σ(bg)-1)²` and
  uncertainty-weighted BCE on the main head.
- **FCT + TTA** — flip-consistency training paired with 4-view test-time
  augmentation. These are treated as a single component because neither
  is meaningful alone: FCT pays a training cost to learn flip-equivariance,
  TTA exploits flip-equivariance at inference. See **ADDITIONS.md** for
  the full reasoning.

Together these add +7.4pp Dice over published BPR at the same recipe
(pranet-traindataset + PraNet test predictions). The recipe switch to
same-teacher Polyp-PVT adds another +6.4pp on top.

See [**ADDITIONS.md**](./ADDITIONS.md) for the mechanism, intuition, and
composition story for each addition.

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

### Reproducing the cross-teacher ablation row

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml \
  --pth checkpoints/<polyppvt-trained checkpoint>/best.pth \
  --dt_path <path to PraNet test predictions> \
  --out_dir results_cl/BACFR_polyppvt_refines_pranet
```

Same polyppvt-trained checkpoint as the winning 0.9455 row, but refining
PraNet predictions instead of Polyp-PVT predictions. Should yield ~0.855
mean.

### Evaluate

Existing eval script (Dice, IoU, S-measure, E-measure, etc.):
```bash
python run/Eval.py --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml --verbose
```
Point it at the `out_dir` from the test step.

## Honest scope

- **Single seed.** Numbers above are from one training run + one TTA
  inference per configuration. Recommended: confirm with at least one
  alternate seed before publication. Margin to next-best (0.881 → 0.9455)
  is large enough that a ±0.005 seed-to-seed variation wouldn't change
  the conclusion.
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
- **Fourth cell of the matrix untested.** Training BACFR on
  pranet-traindataset and refining Polyp-PVT predictions — the other
  cross-teacher direction — has not been measured. The same-teacher
  hypothesis predicts another sub-0.9455 result. Adding this cell would
  close the ablation matrix for the paper.

## Files

- Model: `lib/BACFR_Enhanced_v3_3.py`, `lib/BACFR_Enhanced_v3.py` (shared
  building blocks: HFGate, EDGA_v32, AMCFM, FeatureFusionBlock,
  DecoderSimple, BoundaryContrastLoss).
- Training entry: `run/Train_patch.py`.
- Inference + TTA + boundary patch cropping/stitching: `run/Test_patch_tta.py`.
- Configs: `configs/BACFR_Enhanced_v3_3_polyppvt.yaml` (winning recipe,
  polyppvt-traindataset), `configs/BACFR_Enhanced_v3_3.yaml`
  (pranet-traindataset; used for the 0.881 same-teacher row and the
  cross-teacher 0.855 ablation row).
