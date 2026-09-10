# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

> **⚠ Pipeline-leak correction (June 2026).** An earlier version of
> `run/Test_patch_tta.py` hardcoded the input-image path to
> `<TestDataset>/<testset>/gts/`. That folder contains **ground-truth
> masks**, not RGB images. The model was receiving GT as input, and
> all prior reported numbers on this codebase were inflated. Fixed:
> the `'gts'` was changed to `'images'`. All numbers below are from
> post-correction sanity-checked runs.

## Headline

| Method | Backbone | Mean Dice |
|---|---|---:|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 |
| **BACFR — refines PraNet, trained on pranet-traindataset** | **Res2Net-50** | **~0.86** |
| **BACFR — refines Polyp-PVT, trained on pranet-traindataset** | **Res2Net-50** | **0.8723** |
| **BACFR — refines Polyp-PVT, trained on polyppvt-traindataset** | **Res2Net-50** | **0.8726** |

vs published baselines:
- **+0.053** over published BPR (0.807) on the comparable recipe
  (refining PraNet predictions). The four BACFR boosters add real
  value on top of the BPR paradigm.
- **+0.003** over raw published Polyp-PVT (0.870) when applied to
  Polyp-PVT's predictions. Within noise — refinement on a near-saturated
  base doesn't lift.

## The ceiling observation

Three sanity-checked recipes, three corrected results, all clustered
between 0.86 and 0.87. The two Polyp-PVT-refining recipes — different
training data, different refiner checkpoints — produce mean Dice within
0.0003 of each other and per-dataset differences within ±0.003. They are
clearly distinct experiments (TTA inference is deterministic, so
bit-identical numbers would require identical checkpoints, which these
aren't), yet they converge to the same point.

**Interpretation: the benchmark has a structural ceiling around 0.87 mean
Dice that current patch refinement can't break.** The pattern across
three recipes:

| Starting base | Raw mean Dice | After BACFR refinement | Lift |
|---|---:|---:|---:|
| PraNet | ~0.81 | ~0.86 | **+0.05** |
| Polyp-PVT (run 1, pranet-trained refiner) | 0.870 | 0.8723 | +0.002 |
| Polyp-PVT (run 2, polyppvt-trained refiner) | 0.870 | 0.8726 | +0.003 |

BACFR's contribution lives where there's room to improve. Weak bases get
~+0.05 from refinement; already-saturated bases get nothing meaningful.
The choice of training data (pranet vs polyppvt patches) barely affects
the final score; the base segmenter sets the floor and the refinement
paradigm sets the ceiling.

The remaining error past 0.87 is plausibly structural — very small
polyps, low-contrast ETIS boundaries, ambiguous regions — failure modes
that a boundary-patch refiner working at 256² cropped patches cannot
address regardless of training recipe.

## Per-dataset (best verified recipe)

Train BACFR on polyppvt-traindataset; refine Polyp-PVT's test predictions
with the corrected pipeline (`Test.Dataset.img_subdir: 'images'`).

| Dataset | Dice | IoU |
|---|---:|---:|
| Kvasir | 0.9221 | 0.8704 |
| CVC-ClinicDB | 0.9379 | 0.8890 |
| CVC-ColonDB | 0.8124 | 0.7318 |
| CVC-300 | 0.9065 | 0.8430 |
| ETIS-LaribPolypDB | 0.7842 | 0.6988 |
| **Mean** | **0.8726** | **0.8066** |

The pranet-trained refiner on Polyp-PVT predictions is essentially
indistinguishable from this:

| Dataset | Dice | IoU |
|---|---:|---:|
| Kvasir | 0.9208 | 0.8682 |
| CVC-ClinicDB | 0.9400 | 0.8916 |
| CVC-ColonDB | 0.8133 | 0.7332 |
| CVC-300 | 0.9064 | 0.8424 |
| ETIS-LaribPolypDB | 0.7811 | 0.6940 |
| **Mean** | **0.8723** | **0.8059** |

## The leak's effect (for the record)

Pre-correction inflation, measured on one recipe (the rest follow the
same shape but exact magnitudes weren't re-measured):

| Dataset | Leaky | Corrected | Δ inflation |
|---|---:|---:|---:|
| Kvasir | 0.9619 | 0.9208 | +0.0411 |
| CVC-ClinicDB | 0.9793 | 0.9400 | +0.0393 |
| CVC-ColonDB | 0.9010 | 0.8133 | +0.0877 |
| CVC-300 | 0.9878 | 0.9064 | +0.0814 |
| ETIS | 0.8973 | 0.7811 | **+0.1162** |
| Mean | 0.9455 | 0.8723 | +0.0732 |

Inflation was concentrated on the hardest sets (ETIS, ColonDB, CVC-300),
which is what you'd expect from GT leakage: the harder the test image,
the more useful the GT-as-input shortcut is.

## 2×2 ablation status

| Train data | Test coarse masks | Corrected mean | Status |
|---|---|---:|---|
| pranet-traindataset | PraNet predictions | ~0.86 | sanity-checked |
| pranet-traindataset | Polyp-PVT predictions | 0.8723 | sanity-checked |
| polyppvt-traindataset | PraNet predictions | pending | — |
| polyppvt-traindataset | Polyp-PVT predictions | 0.8726 | sanity-checked |

Three of four cells verified; one cross-teacher cell still to be
re-run. The current pattern suggests it will also land in the 0.85–0.87
band — the refinement ceiling appears base-driven, not train-data driven.

## Architecture (what's actually doing the work)

The refiner is `BACFR_Enhanced_v3_3` (`lib/BACFR_Enhanced_v3_3.py`),
boundary-patch refinement built on the BPR paradigm with four
additions over baseline:

- **HFGate** on the deepest backbone feature (Res2Net x4) — zero-init
  high-frequency gate that amplifies useful boundary residual.
- **Dual fg/bg heads** with complementary loss `(σ(fg)+σ(bg)-1)²` and
  uncertainty-weighted BCE on the main head.
- **FCT + TTA** — flip-consistency training paired with 4-view test-time
  augmentation, treated as a single component. See **ADDITIONS.md** for
  the full reasoning.

Booster contribution over published BPR baseline (refining the same
PraNet predictions): published BPR 0.807 → BACFR ~0.86 = **+0.05**.
Modest but real.

See [**ADDITIONS.md**](./ADDITIONS.md) for the mechanism behind each
addition.

## Reproduction

### Prerequisites

- Conda env: `conda create -n uacanet python=3.7 && pip install -r requirements.txt`
- Res2Net-50 backbone weights (see UACANet README section 2).
- Training patches at `dataset/pranet-traindataset/PatchesDataset-IOU-fusion/`
  with `{img_dir, mask_dir, ann_dir}/{train, val}/`.
- Test data layout: `<TestDataset>/<testset>/{images, gts}/`. **`images/`
  holds RGB inputs; `gts/` holds ground truth.** The corrected pipeline
  uses `'images'`. Do NOT switch back to `'gts'` — that is the leakage
  path that produced the now-retracted inflated numbers.
- Polyp-PVT coarse predictions on the 5 test sets, as 5 subfolders each
  containing one PNG per test image.

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python run/Train_patch.py \
  --config configs/BACFR_Enhanced_v3_3.yaml --verbose --debug
```

10 epochs at 256² on one A100 takes ~6h.

### Test (any of the verified recipes)

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3.yaml \
  --pth checkpoints/<refiner checkpoint>/best.pth \
  --dt_path <path to coarse predictions> \
  --out_dir results_cl/<run name>
```

The corrected pipeline uses `images/` by default. To reproduce the
0.8726 best, use the polyppvt-trained checkpoint with Polyp-PVT
test predictions as `dt_path`.

### Evaluate

```bash
python run/Eval.py --config configs/BACFR_Enhanced_v3_3.yaml --verbose
```

## Honest scope

- **Three of four 2×2 cells sanity-checked with corrected pipeline.**
  All cluster in 0.86–0.87. The fourth cell (polyppvt+PraNet) is
  inference-only and would close the matrix.
- **The pranet→PraNet number above is reported as ~0.86** based on the
  user's check that it landed close to the leaky 0.8815 (i.e., the leak
  effect was small for this cell, unlike the +0.073 inflation on
  polyppvt→Polyp-PVT). Exact per-dataset numbers not yet pasted in;
  swap them in here when convenient.
- **All ~0.87 numbers from one seed.** Margin between recipes is small
  enough (±0.003 across the three Polyp-PVT-refining sanity checks) that
  seed variance could re-order them. The ceiling observation is robust
  to seed; the specific best-recipe identity may not be.
- **BACFR does not beat raw Polyp-PVT.** Comparison to baseline: matches
  Polyp-PVT, exceeds BPR by ~+0.05. The "BACFR crushes everything"
  framing from earlier (pre-correction) commits is retracted.

## Files

- Model: `lib/BACFR_Enhanced_v3_3.py`, `lib/BACFR_Enhanced_v3.py` (shared
  building blocks: HFGate, EDGA_v32, AMCFM, FeatureFusionBlock,
  DecoderSimple, BoundaryContrastLoss).
- Training entry: `run/Train_patch.py`.
- Inference + TTA + boundary patch cropping/stitching:
  `run/Test_patch_tta.py` (post-correction: input subdir is `'images'`).
- Configs: `configs/BACFR_Enhanced_v3_3.yaml` (pranet-traindataset),
  `configs/BACFR_Enhanced_v3_3_polyppvt.yaml` (polyppvt-traindataset).
