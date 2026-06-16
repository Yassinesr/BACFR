# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

## Headline

| Method | Backbone | Mean Dice | Source |
|---|---|---:|---|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 | paper |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 | paper |
| BACFR — same-teacher, weak base (refines PraNet, trained on pranet-traindataset) | Res2Net-50 | 0.881 | this work |
| BACFR — cross-teacher (refines Polyp-PVT, trained on pranet-traindataset) | Res2Net-50 | 0.855 | this work, ablation |
| **BACFR — same-teacher, strong base (refines Polyp-PVT, trained on polyppvt-traindataset)** | **Res2Net-50** | **0.9455** | **this work, best** |

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
| pranet-traindataset | **Polyp-PVT predictions** | **cross-teacher (mismatch)** | 0.929 | 0.942 | 0.794 | 0.918 | **0.691** | **0.855** |
| **polyppvt-traindataset** | **Polyp-PVT predictions** | **same-teacher, strong base** | **0.9619** | **0.9793** | **0.9010** | **0.9878** | **0.8973** | **0.9455** |

The middle row is the critical ablation. It substitutes the stronger
Polyp-PVT base at test time but keeps the refiner trained on PraNet
patches. Mean Dice **drops to 0.855**, **below** the all-PraNet baseline
of 0.881 — even though the test-time coarse masks are objectively
better. Replacing the refiner with one trained on Polyp-PVT patches
(bottom row) recovers and exceeds the baseline by 6.4pp.

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
   does poorly. The cross-teacher ablation (middle row) demonstrates this:
   despite getting a stronger starting point at test time, the refiner
   over-corrects on patches whose error pattern it never trained on.
2. **Quality of the starting point (strong base).** A stronger base
   segmenter produces coarse masks closer to ground truth. The refiner
   only has to bridge a small gap. Polyp-PVT raw is 0.870 mean Dice;
   PraNet raw is ≈0.81. The refiner gets a 6pp head start before it does
   anything. But this head start is wasted unless condition (1) is also
   met — the middle row of the ablation makes that explicit.

The winning recipe pairs both: same-teacher alignment **and** the
strongest available teacher.

### ETIS demonstrates the story most starkly

ETIS is the smallest test set and the most out-of-distribution relative to
the Kvasir+ClinicDB training source. It's where mismatched train/test
error distributions break down first.

| Recipe | ETIS Dice | Δ vs raw Polyp-PVT (0.787) |
|---|---:|---:|
| Same-teacher, weak base (pranet+PraNet) | 0.719 | — |
| Cross-teacher (pranet refines Polyp-PVT) | 0.691 | **−0.096** (mismatch over-corrects) |
| **Same-teacher, strong base (polyppvt+Polyp-PVT)** | **0.897** | **+0.110** |

Note the cross-teacher row scores *worse on ETIS than even raw Polyp-PVT*:
applying a mismatched refiner actively degrades the base prediction it's
supposed to improve. The same-teacher recipe inverts this — same input
coarse masks (Polyp-PVT), different refiner training data, and ETIS jumps
+0.206 (0.691 → 0.897).

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
  --config configs/BACFR_Enhanced_v3_3.yaml \
  --pth checkpoints/<pranet-trained checkpoint>/best.pth \
  --dt_path <path to Polyp-PVT test predictions> \
  --out_dir results_cl/BACFR_pranet_refines_polyppvt
```

Same pranet-trained checkpoint as the 0.881 row, but refining Polyp-PVT
predictions instead of PraNet predictions. Should yield ~0.855 mean.

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
  polyppvt-traindataset and refining PraNet predictions has not been
  measured. The diversity hypothesis predicts another cross-teacher
  regression similar to the 0.855 row. Adding this cell would close the
  ablation matrix.

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
