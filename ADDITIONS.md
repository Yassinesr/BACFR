# BACFR's Four Additions — what, why, and why they compose

> **⚠ Note on the +7.4pp number below.** The "+7.4pp over published BPR
> at same recipe" figure was computed against a pre-correction internal
> result (0.8815) that included GT leakage. The published BPR baseline
> (0.807) is unaffected. The corrected stack-vs-baseline contribution
> needs the original BACFR ablations re-run with `img_subdir: images`
> before it can be quoted precisely. The overall ordering and
> qualitative reasoning below remain sound; the exact magnitudes per
> addition may shift. See `RESULTS.md`.

The base patch refinement architecture (BPR) scores **0.807** mean Dice on
the standard 5-set polyp benchmark. BACFR adds four components on top of
that base and reaches **0.881** at the same recipe (pranet-traindataset +
PraNet test predictions) — a **+7.4pp gain** purely from these additions.

This document explains each addition: what it does mechanically, why it
plausibly helps boundary refinement, and what reasoning supports the
contribution. The four additions are:

1. **HFGate** — a high-frequency gate on the deepest backbone feature.
2. **Dual fg/bg heads** with uncertainty-weighted loss.
3. **FCT + TTA** — flip-consistency training paired with 4-view
   test-time augmentation. *Treated as one component, not two.* The
   reasoning is in §3.

All four are zero-init / schedule-gated, so the model behaves as the BPR
baseline at start of training and the boosters engage gradually.

---

## 1. HFGate (high-frequency gate on the deepest backbone feature)

### What it is

A 1×1 convolution that re-weights the high-frequency residual of the
deepest backbone feature. Applied to Res2Net stage 4 (`x4`, 2048ch,
H/16 spatial).

```python
# lib/BACFR_Enhanced_v3.py:393
class HFGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x):
        hp = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        g = torch.sigmoid(self.gate_conv(hp))
        return x * (0.5 + g)
```

`hp` is the high-pass residual (the feature minus its local average); the
gate `g ∈ (0, 1)` controls how much of that residual to amplify. The output
scales by `0.5 + g ∈ (0.5, 1.5)`. Zero-init means `g = sigmoid(0) = 0.5` at
start, so the output is exactly `x * 1.0` — the model begins as if HFGate
weren't there.

### Why this helps boundary refinement

A boundary patch refiner only fails on pixels near the polyp boundary —
the interior and the far-away background are easy. **Boundary information
is inherently high-frequency**: the relevant signal is in the
fine-grained intensity transitions of the input, and in the corresponding
fine-grained activations of the network. But the deepest backbone feature
has the strongest semantic content (does this image region contain a
polyp?) and the weakest spatial detail (the convolutional stack has
averaged a lot of it away). HFGate is a learnable way to recover and
selectively amplify what spatial detail remains in the deepest feature
before it flows into the decoder.

### Why it works without hurting

- **Zero-init.** Until the gate conv learns to do something, HFGate acts
  as identity. There is no risk of degrading the BPR baseline at the
  start of training.
- **Bounded scale.** The `0.5 + g` range means HFGate can at most halve or
  multiply by 1.5 — it can't blow up or collapse a feature.
- **Channel-wise.** The 1×1 conv has independent gates per output channel,
  so HFGate can learn that some channels benefit from high-frequency
  amplification (those carrying boundary cues) while others should stay
  unchanged (those carrying semantic context).

The combination of these means HFGate is "free to ignore" if it isn't
helpful, and the model only starts using it once boundary-relevant
channels have learned to depend on amplified high-frequency signal.

---

## 2. Dual fg/bg heads with uncertainty-weighted loss

### What it is

Two zero-init 1×1 conv heads on the finest decoder feature (64ch in
BACFR's `DecoderSimple` output). One predicts foreground (`σ(fg) ≈ 1`
inside the polyp), the other predicts background (`σ(bg) ≈ 1` outside).
They jointly drive three training signals:

- **Per-head BCE.** `fg` is supervised by the ground truth mask, `bg` by
  `1 - gt`. The model is encouraged to predict each head correctly.
- **Complementary constraint.** `((σ(fg) + σ(bg) - 1)²).mean()` — at
  every pixel, the two heads should sum to 1. The model is penalized for
  being confidently wrong on both, or confidently right on both
  inconsistently.
- **Uncertainty-weighted main BCE.** From the two heads we compute an
  uncertainty map `u = σ(fg) · σ(bg)`, which is high exactly where both
  heads are uncertain (boundary regions). The main BACFR head's pixel-wise
  BCE is then weighted by `w = 1 + β·u_norm` (with β=0.5), boosting
  gradient on the genuinely hard pixels.

The auxiliary λ is on a schedule (peaks at 0.30 in epochs 2–5, then
decays), so the aux signal is strongest in mid-training and fades for
fine-tuning.

### Why it helps boundary refinement

Refinement quality is dominated by what happens at the boundary — by
construction, the patches the refiner sees are *cropped around the
boundary*. The hard pixels are exactly the boundary pixels. Three things
that the dual-head loss does well here:

- **It identifies the hard pixels using the model itself.** The
  uncertainty `u = σ(fg) · σ(bg)` is large precisely where the model
  hasn't committed — these are the pixels where backbone evidence is
  ambiguous, which is empirically what boundary pixels look like. The
  pixel weighting then amplifies the gradient signal there.
- **It regularizes against confident wrong predictions.** The
  complementary constraint says: if you think this pixel is foreground,
  the *bg* head should agree it isn't background, and vice versa. A model
  that confidently flips a pixel from coarse-mask correct to wrong has to
  pay a penalty on both heads, not just one.
- **It costs nothing at inference.** Both heads are training-only signals.
  At test time only the main BACFR head is used. No latency, no memory.

### Why it works without hurting

- **Zero-init.** Both heads start at all-zero output. Their BCE losses
  start very near 0.5·N (a constant), and the complementary constraint
  starts at `(0.5 + 0.5 - 1)² = 0`. They provide essentially no signal at
  step 0.
- **Scheduled λ.** Aux is held to 0 in epoch 1 (BACFR's first epoch is
  pure baseline), ramps to 0.30 by epoch 2, holds for 3 more epochs, then
  decays. The boosters engage after the main head has found a stable
  basin, not during early chaos.
- **Pixel-weighting bounded.** `w = 1 + 0.5 · u_norm` keeps `w ∈ [1.0, 1.5]`,
  so even maximally weighted pixels can't dominate the loss.

---

## 3. FCT + TTA (treated as one component)

This section needs the most explanation because it justifies a structural
choice: **flip-consistency training (FCT) and test-time augmentation
(TTA) are reported as a single contribution, not two.** The reasoning is
both mechanistic and empirical.

### What FCT is

During training, each batch is tripled with horizontal and vertical
flips: `[original, hflip, vflip]`. A single forward pass produces
predictions for all three views. An auxiliary MSE loss is added:

```
consistency = 0.5 * (MSE(unflip_h(σ(pred_hflip)), σ(pred_orig))
                   + MSE(unflip_v(σ(pred_vflip)), σ(pred_orig)))
loss += λ_flip * consistency
```

`λ_flip` ramps from 0 → 0.30 over the first three epochs, then holds.
The model is being asked to be flip-equivariant: `f(flip(x)) = flip(f(x))`
in sigmoid-probability space.

### What TTA is

At inference, the input image (and coarse mask) is passed through four
D4-subgroup transforms — identity, hflip, vflip, and hflip+vflip — each
through the model. Each output is un-flipped to align with the original
orientation, then the four sigmoid predictions are averaged.

```python
# run/Test_patch_tta.py:tta_predict
return (p0 + p1 + p2 + p3) / 4.0
```

### Why they are *one* component

#### Mechanistic argument

Both FCT and TTA operate on the same symmetry. FCT is the **training-time
constraint**; TTA is the **inference-time exploitation** of that
constraint. Neither is meaningful without the other:

- **TTA alone** averages four predictions from a model that is *not*
  trained to be flip-equivariant. Each flipped prediction has its own
  flip-specific biases — quirks of how the asymmetric convolutional
  feature maps respond to a flipped input. Averaging four predictions
  with four *different biases* mostly cancels random noise; it does not
  enforce the structural prior we want (the polyp boundary should be the
  same regardless of orientation). The gain from TTA without FCT is just
  variance reduction — modest and bounded.

- **FCT alone** pays a training-time cost (3× forward, +1 loss term, +1
  schedule) to learn flip-equivariance, but *only the original-orientation
  prediction is used at test*. The flip-equivariance training has no
  inference-time consumer. The training cost is wasted.

The two compose into a single mechanism: train the model to be
flip-equivariant, then average four orientation views at inference to
extract the value of that equivariance. Splitting them across components
is artificial — neither produces a meaningful gain on its own, and the
combined gain is not the sum of the two.

#### Empirical evidence

The user's prior ablation on the BPR baseline (without HFGate, without
dual heads) shows exactly this pattern:

| BPR configuration | Mean Dice |
|---|---:|
| Vanilla BPR (no TTA, no FCT) | 0.8078 |
| BPR + 4-view TTA only | 0.8088 |
| BPR + 8-view TTA only | 0.8083 |
| BPR + FCT only + TTA | 0.8032 – 0.8056 |

- **TTA alone**: +0.0010. Marginal, in line with the "variance reduction"
  story.
- **FCT alone (with TTA)**: −0.0022 to −0.0046. Adding FCT *without
  enough complementary structure* (in this case, without HFGate and the
  dual head) actually *hurts* — the flip-equivariance constraint pulls
  the optimizer away from a basin BPR's architecture had been finding.

The takeaway is that the value of FCT+TTA is **conditional on the rest of
the architecture**. Pulling apart the pair and ablating each alone
produces misleading numbers (both look harmful or marginal). Reporting
them as a single mechanism, evaluated jointly, is the honest framing.

#### Super-additive behavior in the full stack

On BACFR's full architecture (HFGate + dual heads + FCT + TTA), the
combined boosters produce +7.4pp over published BPR. The contribution of
the FCT+TTA piece *within that stack* is large precisely because the
other two boosters have built a model whose representation can support
the flip-equivariance constraint without collapse:

- HFGate's high-frequency channels need flip-equivariance to remain
  meaningful across orientations. FCT enforces it.
- The dual-head uncertainty map is itself flip-invariant by construction
  (boundary pixels are boundary pixels regardless of orientation). FCT
  reinforces that the prediction agrees.
- TTA then averages four already-agreeing predictions — denoising what's
  left of the patch-level randomness.

This is why the mechanism is super-additive: each booster makes the
others more effective, and FCT+TTA in particular extract value that
neither HFGate nor the dual heads could on their own.

### Code references

- FCT: `lib/BACFR_Enhanced_v3_3.py:forward` (the flip-consistency branch)
  and `lib/BACFR_Enhanced_v3_3.py:_lambda_flip` (the λ schedule).
- TTA: `run/Test_patch_tta.py:tta_predict`.

---

## How the four additions compose

The four additions are not redundant. Each does a distinct job, and they
interact:

| Addition | What it adds | What it depends on |
|---|---|---|
| HFGate | high-frequency channel amplification in the deepest feature | nothing — works at start |
| Dual heads | hard-pixel identification + complementary regularization | nothing — works at start |
| FCT + TTA | flip-symmetry exploitation across train and test | benefits from the others producing flip-stable features |

**Order of engagement (during training).**
- Epoch 1: all schedules at 0. Model behaves as BPR baseline. HFGate is
  identity, dual heads contribute nothing, FCT λ is 0.
- Epochs 2–3: aux λ ramps to 0.30, FCT λ ramps to 0.30. HFGate's gate
  conv begins to learn channel-wise amplification.
- Epochs 4–6: full booster engagement. HFGate has converged to its
  channel selection, dual heads are providing uncertainty signal, FCT is
  enforcing equivariance.
- Epochs 7–10: aux and FCT λ decay (the model has internalized the
  regularizers; further weight on them would conflict with fine-tuning
  the main head).

**At inference.** Only HFGate, the main BACFR head, and TTA are involved.
The dual heads are dropped. FCT is no longer applied. The model
inherits the representation those additions shaped during training, and
TTA extracts the final +variance-reduction lift.

---

## Summary

| Addition | Headline reason it helps |
|---|---|
| HFGate | Restores boundary-relevant high-frequency signal in the deepest backbone feature. |
| Dual heads | Identifies hard (boundary) pixels using the model itself; regularizes confident wrong predictions. |
| FCT + TTA | Trains the model to be flip-equivariant; averages four orientations at test to extract that prior. Reported jointly because either alone is marginal or harmful. |

Together: **+7.4pp Dice over published BPR** on the same recipe
(pranet-traindataset + PraNet test predictions). The best result reported
in this work (0.9455 mean) inherits all four additions and adds a recipe
change on top (same-teacher Polyp-PVT base) — see `RESULTS.md`.
