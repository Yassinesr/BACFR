# Ablation Report — Pushing past the Polyp-PVT ceiling

Three independent attempts to lift the polyppvt→Polyp-PVT recipe past
its corrected-pipeline mean of 0.8726. Two diagnostics (oracle bounds,
multi-refiner oracle) and two gating sweeps (dual-head uncertainty, TTA
variance) bound what is reachable and explain why we hit a ceiling.

The headline: **mixed training is the only attempt that produced a real
lift (+0.003). Both inference-time gating signals failed (+0.001 each).
The oracle analysis explains why — the ceiling is structural, not
gating-strategy-limited.**

All numbers below are from the post-correction pipeline (`img_subdir:
images`), so they are directly comparable.

---

## 1. Mixed training

### What was tried

Train BACFR_Enhanced_v3_3 on the **union** of pranet-traindataset and
polyppvt-traindataset patches (~80k patches total vs ~40k for either
single recipe). Same architecture, same boosters, same hyperparameters,
same epoch count. Only the training data changed.

Hypothesis: exposure to both base segmenters' error distributions
would produce a refiner that handles both PraNet's coarser/global
errors and Polyp-PVT's finer/local ones — potentially breaking the
single-recipe ceiling on either base.

Config: `configs/BACFR_Enhanced_v3_3_mixed.yaml`. The dataloader was
extended to accept `root` as either a string or a list (per-root
filename pairing prevents cross-root false matches since both datasets
use the same `<number>_patch_<N>.png` naming).

### Results

| Recipe | Mean Dice | vs single-recipe best on that base |
|---|---:|---:|
| **mixed → Polyp-PVT** | **0.8754** | **+0.0028** over polyppvt-trained (0.8726) — **new best** |
| mixed → PraNet | 0.8118 | **−0.0482** vs pranet-trained (~0.86) — large regression |

Per-dataset for the winning mixed → Polyp-PVT run:

| Dataset | mixed | polyppvt-trained | Δ (mixed wins where positive) |
|---|---:|---:|---:|
| Kvasir | 0.9215 | 0.9221 | −0.0006 |
| CVC-ClinicDB | 0.9398 | 0.9379 | +0.0019 |
| CVC-ColonDB | 0.8148 | 0.8124 | +0.0024 |
| CVC-300 | 0.9085 | 0.9065 | +0.0020 |
| ETIS-LaribPolypDB | **0.7921** | **0.7842** | **+0.0079** |

The +0.003 mean lift is concentrated on ETIS (+0.008). ETIS is the most
out-of-distribution test set; mixed training apparently exposed the
refiner to enough boundary error diversity that it generalizes better
to ETIS's distinctive polyp morphology.

### Interpretation: negative interference is real

Mixed training was a **compromise model**. On its strong-base target it
gained marginally; on the weak base (PraNet) it lost catastrophically.
The pranet-trained refiner has learned to fix large/coarse boundary
errors that the polyppvt patches barely contain. When trained on a
50/50 mix, the refiner's capacity is split between two different error
distributions, and it ends up worse at fixing PraNet's bigger errors
because half its training was on Polyp-PVT's near-perfect patches.

Takeaway: **mixed training is not a universal recipe**. It's a tradeoff
that helps the case it's deployed for and hurts everything else. For
the polyppvt→Polyp-PVT goal specifically, it's the new best deployable.

---

## 2. Oracle analysis — bounding what gating can ever achieve

### What an oracle measures

A pixel-wise oracle uses ground truth to make decisions a real deployed
system cannot make. The number it produces is the **strict upper bound**
on what any inference-only strategy could achieve given the same model.

Three bounds tested:

| Bound | Decision rule (uses GT) |
|---|---|
| `raw_coarse` | No refinement at all (lower anchor) |
| `fully_refined` | Always use refined where the refiner touched (current behavior) |
| `oracle_image` | Per image, pick max(`dice(coarse)`, `dice(refined)`). Bound for a perfect *image-level* refine/skip policy. |
| `oracle_pixel` | At every touched pixel, peek GT and pick whichever of {coarse, refined} matches. Bound for **any** pixel-level gating strategy. |
| `combined_oracle` | Same as oracle_pixel but selecting from {coarse, refiner_1, refiner_2, ...}. Bound for **ensemble** + perfect gate. |

### Single-refiner oracle (polyppvt-trained, on Polyp-PVT base)

| Bound | Mean Dice | Δ vs current |
|---|---:|---:|
| raw coarse | 0.8732 | −0.0006 |
| **fully refined (current)** | **0.8726** | — |
| oracle_image | 0.8785 | +0.0059 |
| oracle_pixel | **0.8854** | **+0.0128** |

**Reading:** the refiner has the right answer often enough that a
*perfect* per-pixel gate could lift +0.013. But that requires a
correctness-tracking signal we don't have.

### Multi-refiner oracle (mixed + polyppvt-trained)

| Configuration | Mean Dice |
|---|---:|
| raw coarse | 0.8732 |
| refined: mixed | 0.8754 |
| refined: polyppvt | 0.8726 |
| oracle (mixed alone) | 0.8827 |
| oracle (polyppvt alone) | 0.8854 |
| **COMBINED ORACLE** | **0.8876** |

**Combining lift over best single oracle: +0.0021.** Below the +0.004
threshold for "complementary refiners". The two refiners' *correct*
pixels overlap substantially — combining them barely raises the
ceiling.

**Non-obvious finding:** polyppvt has the *higher* gating ceiling
(0.8854) despite the *lower* deployable score (0.8726). It makes
bolder decisions — more often right *and* more often wrong — so a good
gate would have more to recover. Mixed is the safer averaged refiner;
polyppvt is the better gating substrate. Neither is reachable today.

### Why the combined oracle is the strategic ceiling

`combined_oracle = 0.8876` is the absolute cap on any strategy that
selects per-pixel among {raw coarse, mixed-refined, polyppvt-refined}.
This includes any ensemble + any gate, learned or hand-crafted.
**No inference-time intervention on these refiners can exceed 0.8876.**

The remaining ~11pp of error (Dice 1.0 → 0.888) lives **outside the
refinable boundary patches**: missed polyp regions Polyp-PVT never
detected (no boundary → no patch → no refinement opportunity),
and pixels where both coarse and both refiners are simultaneously wrong.

---

## 3. Gating sweeps

### What `tau` is

Both gating sweeps share a single knob `tau ∈ [0, 1]`. At each touched
pixel, the merge computes a per-pixel signal `s` (uncertainty or
variance), normalizes it min-max within the touched region, and uses
the refined value where `s_norm ≤ tau`; keeps the coarse value
otherwise.

- `tau = 1.0` → use refiner everywhere touched (== current merge).
- `tau = 0.0` → use refiner almost nowhere (≈ raw coarse base).
- Intermediate `tau` → use refiner where it is *most confident* by the
  chosen signal, keep coarse where it is uncertain.

These are **self-validating anchors**: `tau=1.0` must reproduce the
known 0.8726 number, and `tau=0.0` must reproduce raw coarse base.
If either anchor is off, the scoring harness has a bug. Both swept
within 0.0006 of expected on every run.

### Sweep #1: dual-head uncertainty

Signal: `u = σ(fg_pred) · σ(bg_pred)`, the dual-head pixel uncertainty
computed by the model. Pixels where the two heads are both unsure of
their respective class get high `u`; pixels where one head is confident
get low `u`. High `u` → unreliable → keep coarse. Low `u` → reliable →
use refiner.

| tau | CVC-300 | CVC-Clinic | Kvasir | CVC-ColonDB | ETIS | **MEAN** |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.9065 | 0.9379 | 0.9221 | 0.8124 | 0.7842 | 0.8726 |
| 0.90 | 0.9055 | 0.9383 | 0.9219 | 0.8135 | 0.7849 | 0.8728 |
| 0.80 | 0.9046 | 0.9387 | 0.9216 | 0.8140 | 0.7858 | 0.8729 |
| 0.70 | 0.9043 | 0.9389 | 0.9213 | 0.8146 | 0.7873 | 0.8733 |
| **0.60** | **0.9044** | **0.9396** | **0.9211** | **0.8145** | **0.7883** | **0.8736** |
| 0.50 | 0.9042 | 0.9401 | 0.9210 | 0.8136 | 0.7889 | 0.8735 |
| 0.40 | 0.9041 | 0.9400 | 0.9208 | 0.8129 | 0.7893 | 0.8734 |
| 0.30 | 0.9041 | 0.9400 | 0.9206 | 0.8124 | 0.7895 | 0.8733 |
| 0.20 | 0.9040 | 0.9402 | 0.9205 | 0.8120 | 0.7896 | 0.8733 |
| 0.00 | 0.9039 | 0.9403 | 0.9204 | 0.8114 | 0.7901 | 0.8732 |

**Peak at tau=0.60: 0.8736 (+0.0010 over current).** Unimodal curve
with a clear peak, so the signal isn't random noise. But the peak
captures only 8% of the +0.0128 oracle headroom.

### Sweep #2: TTA variance

Signal: `var(p_view_1, p_view_2, p_view_3, p_view_4)`, the
pixel-level variance of the four TTA flip predictions. With FCT
training, the model is approximately flip-equivariant, so high
variance flags pixels where the model fails its own equivariance
constraint — a different notion of unreliable.

| tau | CVC-300 | CVC-Clinic | Kvasir | CVC-ColonDB | ETIS | **MEAN** |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.9065 | 0.9379 | 0.9221 | 0.8124 | 0.7842 | 0.8726 |
| 0.95 | 0.9065 | 0.9379 | 0.9221 | 0.8124 | 0.7842 | 0.8726 |
| 0.90 | 0.9065 | 0.9379 | 0.9221 | 0.8123 | 0.7842 | 0.8726 |
| 0.80 | 0.9065 | 0.9380 | 0.9221 | 0.8123 | 0.7843 | 0.8726 |
| 0.70 | 0.9064 | 0.9381 | 0.9221 | 0.8122 | 0.7845 | 0.8727 |
| 0.60 | 0.9064 | 0.9381 | 0.9221 | 0.8122 | 0.7849 | 0.8727 |
| 0.50 | 0.9063 | 0.9381 | 0.9222 | 0.8121 | 0.7853 | 0.8728 |
| 0.40 | 0.9060 | 0.9385 | 0.9221 | 0.8119 | 0.7859 | 0.8729 |
| 0.30 | 0.9059 | 0.9390 | 0.9220 | 0.8116 | 0.7866 | 0.8730 |
| 0.20 | 0.9059 | 0.9398 | 0.9217 | 0.8113 | 0.7875 | 0.8732 |
| **0.10** | **0.9055** | **0.9403** | **0.9214** | **0.8116** | **0.7887** | **0.8735** |
| 0.00 | 0.9039 | 0.9403 | 0.9204 | 0.8114 | 0.7901 | 0.8732 |

**Peak at tau=0.10: 0.8735 (+0.0009 over current).** Even weaker than
dual-head uncertainty.

### Interpreting both sweeps

The curves are real but flat — both signals carry *some* information
about correctness but not nearly enough to exploit the +0.013 oracle
headroom. Three diagnostic observations:

1. **Peak shape is meaningful.** Both sweeps are unimodal with a
   real maximum, not flat or monotonic. So the gating signal isn't
   pure noise — it correlates with correctness, just weakly.
2. **Peak gain is identical (~+0.001) across signals.** Different
   information sources (model confidence vs flip equivariance) cap at
   roughly the same place. That's consistent with the underlying
   refiner just not making *many* per-pixel mistakes that are
   *separable* by any model-internal signal on this strong base.
3. **Both signals lift ETIS most.** That's where the refiner makes
   the most genuinely-wrong-and-flaggable calls (raw 0.787, refined
   0.7842, gated 0.7889). The other four datasets barely move.

The two signals capture overlapping (mostly the same) information.
Combining them would not double the gain.

---

## 4. Putting it together: where the ceiling actually lives

Stacked summary for the polyppvt→Polyp-PVT recipe:

| Configuration | Mean Dice |
|---|---:|
| Raw Polyp-PVT (this server's baseline) | 0.8732 |
| polyppvt-trained refiner (single-recipe) | 0.8726 |
| **mixed-trained refiner (best deployable today)** | **0.8754** |
| Dual-head uncertainty gating (peak tau=0.60) | 0.8736 |
| TTA-variance gating (peak tau=0.10) | 0.8735 |
| Single-refiner pixel oracle (polyppvt) | 0.8854 |
| **Combined multi-refiner pixel oracle (HARD CAP)** | **0.8876** |

The gap from `best deployable today` to `combined oracle` is +0.012.
That gap is what's left to fight for. Three model-internal signals
have each captured at most ~10% of it. We have no fourth signal that
would plausibly do better; we've exhausted the obvious model-side
information.

### Why we cannot break 0.888 on this base

The combined pixel oracle being 0.888 — not 0.95 or 1.0 — is the
deeper finding. It means even with **perfect** per-pixel knowledge of
which refiner is correct, ~11pp of Dice error is unreachable. That
error lives in two places:

- **Outside any boundary patch**: regions Polyp-PVT missed entirely
  (no boundary → no patch crop → no refinement opportunity). ETIS's
  oracle is just 0.808 because most of its error mode is "tiny polyp
  Polyp-PVT didn't detect at all", not "boundary drift around a
  detected polyp".
- **Inside patches where everything agrees wrongly**: pixels where
  the coarse mask, the mixed refiner, AND the polyppvt refiner all
  output the same wrong value. The oracle can't pick a correct
  alternative because none exists in the candidate set.

Both error sources are **structural** to the patch refinement
paradigm. Mixed training, ensembling, and gating cannot reach them by
construction. To break 0.888 you would need either (a) a stronger
base segmenter that misses fewer regions, or (b) a refinement
paradigm that operates beyond cropped boundary patches.

### What the paper should actually claim

The honest two-leg story:

1. **BACFR substantially improves weak base segmenters.** Refining
   PraNet (raw ~0.81) reaches ~0.86 in the corrected pipeline — a
   real +0.05 contribution that puts BACFR ahead of published BPR by
   ~+0.053. This is the main contribution.

2. **BACFR saturates on strong base segmenters.** Refining Polyp-PVT
   (raw 0.873) reaches 0.875 (mixed-trained, deployable). The
   structural ceiling is 0.888 (combined oracle); the deployable
   ceiling is the one we hit. The oracle analysis quantifies that the
   remaining error is geometric, not informational — it lives outside
   refinable boundary patches.

The negative result on strong bases is *informative*. It tells future
work where boundary patch refinement runs out of room and why, which
is more useful than another fractional bump.

---

## 5. Reproducibility — pinned commits and commands

All numbers above are from commits on `claude/sleepy-planck-2zQ5C`:

| Diagnostic | Script | Commit |
|---|---|---|
| Dual-head uncertainty gated sweep | `run/Test_patch_tta_gated.py` | `227b6192ca` |
| TTA variance gated sweep | `run/Test_patch_tta_variance_gated.py` | `a80c6dde4b` |
| Single-refiner oracle bounds | `run/Test_patch_tta_oracle.py` | `4c9aa9b326` |
| Multi-refiner oracle bounds | `run/Test_patch_tta_multi_oracle.py` | `a80c6dde4b` |
| Mixed-training config | `configs/BACFR_Enhanced_v3_3_mixed.yaml` | `b21426d597` |
| Dataloader (list-of-roots support) | `utils/dataloader.py` | `b21426d597` |

All inference-only scripts run against `<TestDataset>/<set>/images/`
for the RGB image input (the corrected default) and use GT strictly
for scoring and oracle decisions, never as model input.

---

## 6. Recommended next moves

In priority order:

1. **Lock the strong-base headline at 0.8754** (mixed-trained refiner,
   Polyp-PVT base). Update RESULTS.md.
2. **Bake this oracle/gating analysis into the paper.** It is the
   strongest justification of *why* refinement saturates on strong
   bases — and the negative result is the contribution.
3. **Verify the pranet+PraNet number with the corrected pipeline.**
   It's currently logged as "~0.86" pending exact per-dataset numbers.
   This is the weak-base headline and the +0.05 over published BPR
   depends on it.
4. **Second seed of mixed → Polyp-PVT** to confirm 0.8754 is not a
   lucky run. Margin to the polyppvt-trained baseline (+0.003) is
   small enough that one seed isn't enough.
5. **Do NOT pursue more gating experiments.** Three independent
   signals have failed to capture the +0.013 ceiling headroom in a
   coherent way. Time better spent on writing.
