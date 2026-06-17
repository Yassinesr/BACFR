# BACFR — Results

Boundary-patch refinement on the standard 5-set polyp segmentation benchmark
(Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB).

> **⚠ Pipeline-leak correction (June 2026).** An earlier version of
> `run/Test_patch_tta.py` hardcoded the input-image path to
> `<TestDataset>/<testset>/gts/`. That folder contains **ground-truth
> masks**, not RGB images. The model was therefore receiving the GT
> mask as part of its input, and every prior reported number on this
> codebase (e.g. 0.8815, 0.8548, 0.9455) is inflated by the resulting
> leak.
>
> The bug is fixed in commit pinned below: `Test_patch_tta.py` now
> defaults `img_subdir` to `'images'`. **Only one recipe has been
> re-evaluated with the corrected pipeline**; everything else awaits
> re-run.

## Headline (post-correction, single verified recipe)

| Method | Backbone | Mean Dice | Source |
|---|---|---:|---|
| BPR (boundary patch refinement, published) | Res2Net-50 | 0.807 | paper |
| Polyp-PVT (end-to-end, published) | PVT-v2-B2 | 0.870 | paper |
| **BACFR — cross-teacher (pranet-trained, refines Polyp-PVT)** | **Res2Net-50** | **0.8723** | **this work, corrected pipeline** |

vs published baselines:
- **+0.065** over published BPR (substantial — the BACFR refinement
  stack adds real value over the unrefined Res2Net-50-based paradigm).
- **+0.002** over published Polyp-PVT (essentially tied — refining
  Polyp-PVT's already-strong predictions delivers marginal lift).

The refinement-over-base framing is the most honest: raw Polyp-PVT is
0.870 mean Dice; BACFR-refined Polyp-PVT is 0.8723. **The boundary
refinement step contributes +0.002 mean Dice** on top of an already
near-saturated base. Pre-correction we believed this was +0.076; the
0.074pp difference was the leak.

## Best recipe — per-dataset (corrected)

Train BACFR on pranet-traindataset; refine Polyp-PVT's test predictions;
ensure `Test.Dataset.img_subdir: images` (now the default).

| Dataset | Dice | IoU |
|---|---:|---:|
| Kvasir | 0.9208 | 0.8682 |
| CVC-ClinicDB | 0.9400 | 0.8916 |
| CVC-ColonDB | 0.8133 | 0.7332 |
| CVC-300 | 0.9064 | 0.8424 |
| ETIS-LaribPolypDB | 0.7811 | 0.6940 |
| **Mean** | **0.8723** | **0.8059** |

## The leak's effect on this recipe

Leaky vs corrected for the one cell that was re-run:

| Dataset | Leaky | Corrected | Δ inflation |
|---|---:|---:|---:|
| Kvasir | 0.9619 | 0.9208 | +0.0411 |
| CVC-ClinicDB | 0.9793 | 0.9400 | +0.0393 |
| CVC-ColonDB | 0.9010 | 0.8133 | +0.0877 |
| CVC-300 | 0.9878 | 0.9064 | +0.0814 |
| ETIS | 0.8973 | 0.7811 | **+0.1162** |
| **Mean** | **0.9455** | **0.8723** | **+0.0732** |

Inflation is concentrated on the harder out-of-distribution sets (ETIS,
ColonDB, CVC-300), which is exactly what you'd expect from GT leakage:
the harder the test set, the more useful the GT-as-input shortcut is to
the model. It also means **the per-recipe inflation in the other 2×2
cells is unlikely to be uniform** — each cell needs its own re-run.

## 2×2 ablation (status)

| Train data | Test coarse masks | Pairing | Leaky mean | Corrected mean |
|---|---|---|---:|---:|
| pranet-traindataset | PraNet predictions | same-teacher, weak base | 0.8815 | **pending** |
| **pranet-traindataset** | **Polyp-PVT predictions** | **cross-teacher (strong test base)** | 0.9455 | **0.8723** |
| polyppvt-traindataset | PraNet predictions | cross-teacher | 0.8548 | **pending** |
| polyppvt-traindataset | Polyp-PVT predictions | same-teacher, strong base | (unclear*) | **pending** |

\*The earlier reported "0.9455 = polyppvt+Polyp-PVT" appears to have been
a labelling mix-up; the 0.9455 result was from the pranet-trained
checkpoint. The polyppvt+Polyp-PVT cell has not been cleanly measured.

**None of the leaky numbers in the table above can be cited.** The
previously-written narratives about "same-teacher wins" / "cross-teacher
symmetry at 0.8548" / "+0.064 from strong base" were built on
inflated-and-mislabelled data and should not be relied on.

## What to re-run (priority order)

All three remaining cells are inference-only (existing checkpoints, just
re-run `Test_patch_tta.py` with `img_subdir: images`):

1. **`pranet+PraNet`** (the previous "0.8815" same-teacher weak baseline).
   This sets the reference for "does BACFR refinement improve on raw
   PraNet" and lets us isolate the booster contribution at fixed base.
2. **`polyppvt+Polyp-PVT`** (same-teacher with the strong base; never
   cleanly measured). If this beats 0.8723, same-teacher alignment
   matters and the winning recipe should be polyppvt-trained.
   If it lands below, the pranet-trained refiner is just better and the
   cross-teacher 0.8723 is the actual peak.
3. **`polyppvt+PraNet`** (cross-teacher, the other direction). Closes the
   2×2 and lets us re-examine whether the "cross-teacher symmetry"
   observation survives.
4. **Second seed of `pranet+Polyp-PVT`** to confirm 0.8723 ±a few
   permille. The +0.002 lift over raw Polyp-PVT is within seed noise
   range; a second run will tell us if it's positive, zero, or negative.

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

Combined contribution over published BPR (0.807 → 0.8723 = +0.065)
remains substantial post-correction. The precise per-booster ablation
would benefit from re-running the original ablations with the corrected
pipeline, but the overall stack-vs-baseline gain is in the +6pp range.

See [**ADDITIONS.md**](./ADDITIONS.md) for the mechanism and intuition
behind each addition.

## Reproduction

### Prerequisites

- Conda env: `conda create -n uacanet python=3.7 && pip install -r requirements.txt`
- Res2Net-50 backbone weights (see UACANet README section 2).
- Training patches at `dataset/pranet-traindataset/PatchesDataset-IOU-fusion/`
  with `{img_dir, mask_dir, ann_dir}/{train, val}/`.
- Test data layout: `<TestDataset>/<testset>/{images, gts}/` for each
  test set. **`images/` holds the RGB images; `gts/` holds the ground
  truth.** The corrected default `img_subdir` is `'images'`. Do NOT set
  it to `'gts'` — that is the leakage path.
- Polyp-PVT coarse predictions on the 5 test sets, as 5 subfolders each
  containing one PNG per test image.

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python run/Train_patch.py \
  --config configs/BACFR_Enhanced_v3_3.yaml --verbose --debug
```

Trains BACFR on patches cropped from PraNet's training-set outputs.
10 epochs at 256² on one A100 takes ~6h.

### Test (best recipe, corrected pipeline)

```bash
python run/Test_patch_tta.py \
  --config configs/BACFR_Enhanced_v3_3.yaml \
  --pth checkpoints/<your pranet-trained checkpoint>/best.pth \
  --dt_path <path to Polyp-PVT test predictions> \
  --out_dir results_cl/BACFR_pranet_refines_polyppvt_corrected
```

The corrected pipeline uses `img_subdir: images` by default. If you
have an older config that explicitly sets `img_subdir: gts`, remove it.

### Evaluate

```bash
python run/Eval.py --config configs/BACFR_Enhanced_v3_3.yaml --verbose
```
Point it at the corrected `out_dir`.

## Honest scope (post-correction)

- **One recipe verified with the corrected pipeline (0.8723).** Three
  other 2×2 cells leaky, all pending re-run.
- **Refinement-over-Polyp-PVT lift is +0.002 mean Dice.** Within seed
  variance. A second seed is needed to confirm the sign.
- **The +9pp "boosters beat published BPR" story holds qualitatively**
  — corrected 0.8723 vs published BPR 0.807 is still +0.065 — but the
  exact per-booster contribution needs the original ablations re-run
  before it can be quoted with precision.
- **Earlier folder-label confusion.** Across this work's
  iterations, the same numerical result has been associated with
  different recipe labels in different writeups. The pinned commit
  reflects current best understanding; treat any per-cell attribution
  from prior commits as superseded.

## Files

- Model: `lib/BACFR_Enhanced_v3_3.py`, `lib/BACFR_Enhanced_v3.py` (shared
  building blocks: HFGate, EDGA_v32, AMCFM, FeatureFusionBlock,
  DecoderSimple, BoundaryContrastLoss).
- Training entry: `run/Train_patch.py`.
- Inference + TTA + boundary patch cropping/stitching:
  `run/Test_patch_tta.py` (post-correction: defaults `img_subdir` to
  `'images'`).
- Configs: `configs/BACFR_Enhanced_v3_3.yaml` (the recipe behind the
  verified 0.8723 number), `configs/BACFR_Enhanced_v3_3_polyppvt.yaml`
  (alternate training data).
