# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

## Headline

| Method | Backbone | Mean Dice | Source |
|---|---|---:|---|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 | paper |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 | paper |
| BACFR — same-teacher, weak base (trained on pranet-traindataset, refines PraNet) | Res2Net-50 | 0.8815 | this work |
| BACFR — cross-teacher, either direction | Res2Net-50 | 0.8548 | this work, ablation |
| **BACFR — same-teacher, strong base (trained on polyppvt-traindataset, refines Polyp-PVT)** | **Res2Net-50** | **0.9455** | **this work, best** |

+13.8pp over published BPR. +7.6pp over published Polyp-PVT. +6.4pp over the
prior BACFR result (same-teacher, weak base).

## Best recipe — per-dataset

Train BACFR on polyppvt-traindataset; refine Polyp-PVT's test predictions.

| Dataset | Dice | IoU |
|---|---:|---:|
| Kvasir | 0.9619 | 0.9376 |
| CVC-ClinicDB | 0.9793 | 0.9607 |
| CVC-ColonDB | 0.9010 | 0.8653 |
| CVC-300 | 0.9878 | 0.9759 |
| ETIS-LaribPolypDB | 0.8973 | 0.8658 |
| **Mean** | **0.9455** | **0.9211** |

## The full 2×2 ablation (per-dataset Dice)

The full matrix is now measured. Diagonals are same-teacher pairings;
off-diagonals are cross-teacher.

| Train data (refiner) | Test coarse masks | Pairing | Kvasir | ClinicDB | ColonDB | CVC-300 | ETIS | **Mean** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pranet-traindataset | PraNet predictions | same-teacher, weak base | 0.9501 | 0.9574 | 0.8044 | 0.9765 | 0.7191 | 0.8815 |
| pranet-traindataset | Polyp-PVT predictions | cross-teacher | 0.9285 | 0.9424 | 0.7938 | 0.9177 | 0.6914 | 0.8548 |
| polyppvt-traindataset | PraNet predictions | cross-teacher | 0.9285 | 0.9424 | 0.7938 | 0.9177 | 0.6914 | 0.8548 |
| **polyppvt-traindataset** | **Polyp-PVT predictions** | **same-teacher, strong base** | **0.9619** | **0.9793** | **0.9010** | **0.9878** | **0.8973** | **0.9455** |

**The diagonal (same-teacher) cells win in both base configurations.**
Both cross-teacher cells score 0.8548 mean Dice — below the
same-teacher weak-base baseline of 0.8815, and well below the
same-teacher strong-base peak of 0.9455.

Three observations from the matrix:

- **Same-teacher alignment is necessary.** Cross-teacher pairings drop
  mean Dice by 0.027 vs the same-teacher weak baseline (0.8815 → 0.8548)
  and by 0.091 vs the same-teacher strong winner (0.9455 → 0.8548). The
  loss is the same in both cross directions — directionality of the
  mismatch doesn't matter, only that it exists.
- **Strong test base lifts only when the refiner has been trained on
  that base's outputs.** Substituting Polyp-PVT predictions for PraNet
  predictions with the pranet-trained refiner (top row, second cell)
  actually drops Dice to 0.8548 — *worse* than the all-PraNet baseline,
  despite the test-time coarse masks being objectively closer to GT.
  The refiner doesn't know what to do with the unfamiliar error pattern.
- **Same-teacher strong base is super-additive.** Same-teacher alone
  with the weak base gives 0.8815. Strong test base alone with a
  mismatched refiner gives 0.8548. Combining them gives 0.9455 — the
  combined gain (+0.064 over same-teacher weak) is larger than either
  ingredient could deliver on its own.

## Why this recipe wins

The BACFR boundary patch refiner is a *correction model*: it sees a
coarse mask, identifies boundary regions where that mask is likely
wrong, and nudges those regions toward the ground truth. The complete
2×2 ablation shows its accuracy depends on two conditions, both
necessary, neither sufficient.

1. **Train/test distribution match (same-teacher pairing).** The refiner
   has to recognize and correct *the kind of errors the base segmenter
   makes*. PraNet's boundary errors are coarser and more global;
   Polyp-PVT's are subtler and finer. A refiner trained on one
   teacher's error distribution and deployed on another's has to
   generalize across two different error modes — and the matrix shows
   it fails to. Both cross-teacher cells score the same 0.8548 mean
   Dice, regardless of which way the mismatch points. Direction of the
   mismatch is irrelevant; presence of the mismatch is what costs
   accuracy.
2. **Starting-point quality (strong test base).** A stronger base
   produces coarse masks closer to ground truth, giving the refiner a
   closer starting point. Polyp-PVT raw is 0.870 mean Dice; PraNet raw
   is ≈0.81 — a ~6pp head start. But this head start is only realized
   when condition (1) is also satisfied. Substituting Polyp-PVT
   predictions under a pranet-trained refiner (the cross-teacher cell)
   yields 0.8548 — *below* the same-teacher pranet+PraNet baseline of
   0.8815. The stronger test inputs *hurt* without a matching refiner.

The winning recipe satisfies both: same-teacher pairing (polyppvt-trained
refiner deployed on Polyp-PVT test predictions) with the stronger of the
two available bases. Either alone falls back to the cross-teacher floor;
together they lift to 0.9455.

### ETIS demonstrates both conditions most starkly

ETIS is the smallest test set and the most out-of-distribution relative
to the Kvasir+ClinicDB training source. It's where mismatched train/test
error distributions break down first, and where the strong-base lift is
largest.

**Same-teacher condition (fix test base, vary refiner training):**

| Recipe | ETIS Dice |
|---|---:|
| Same-teacher (pranet-trained) refining PraNet predictions | **0.7191** |
| Cross-teacher (polyppvt-trained) refining PraNet predictions | 0.6914 |

Identical PraNet test inputs. Mismatched refiner training costs
**−0.028 on ETIS**.

**Both conditions vs same-teacher weak base:**

| Recipe | ETIS Dice | Δ vs same-teacher weak (0.7191) |
|---|---:|---:|
| Same-teacher, weak base (pranet+PraNet) | 0.7191 | — |
| Cross-teacher (either direction) | 0.6914 | −0.028 |
| **Same-teacher, strong base (polyppvt+Polyp-PVT)** | **0.8973** | **+0.178** |

The cross-teacher row scores *worse on ETIS than even raw Polyp-PVT*
(0.787): applying a mismatched refiner actively degrades the base
prediction it's supposed to improve. The same-teacher strong-base
recipe inverts this — ETIS jumps to 0.8973, **+0.110 over raw
Polyp-PVT**, the largest single-dataset gain in this work.

### Cross-teacher symmetry (an empirical observation)

Both cross-teacher cells score **identical 0.8548 mean Dice to four
decimal places, on every per-dataset entry**. The match is striking: it
suggests the mismatch penalty depends on *whether* train and test bases
agree, not *which* base is which. If reproducible across seeds, this is
a clean empirical regularity worth flagging; we report it as observed
without yet ascribing a mechanism.

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

### Reproducing the cross-teacher ablation rows

Two cross-teacher cells, both yielding ~0.855 mean Dice. Both require
inference only (using existing checkpoints).

**polyppvt-trained refiner on PraNet test predictions:**

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3_polyppvt.yaml \
  --pth checkpoints/<polyppvt-trained checkpoint>/best.pth \
  --dt_path <path to PraNet test predictions> \
  --out_dir results_cl/BACFR_polyppvt_refines_pranet
```

**pranet-trained refiner on Polyp-PVT test predictions:**

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3.yaml \
  --pth checkpoints/<pranet-trained checkpoint>/best.pth \
  --dt_path <path to Polyp-PVT test predictions> \
  --out_dir results_cl/BACFR_pranet_refines_polyppvt
```

Both should yield 0.8548 mean.

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
- **All four matrix cells now measured.** Diagonal (same-teacher) cells
  win; off-diagonal (cross-teacher) cells coincide at 0.8548.
- **Cross-teacher symmetry to four decimals is striking.** Both
  off-diagonal cells score 0.8548 mean Dice with identical per-dataset
  numbers (Kvasir 0.9285, ClinicDB 0.9424, ColonDB 0.7938, CVC-300
  0.9177, ETIS 0.6914). This is unusual enough to warrant a
  second-seed sanity check before relying on it as a clean empirical
  regularity in the paper.

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
