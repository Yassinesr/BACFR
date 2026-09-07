#!/usr/bin/env bash
# =============================================================================
#  Flip-augmentation vs FCT+TTA ablation ladder  --  runs on 2 GPUs in parallel
# =============================================================================
#
#  Hypothesis: random_flip/random_rotate and FCT+TTA are REDUNDANT mechanisms.
#  Augmentation already installs the orientation invariance, so the consistency
#  loss has no gradient signal left and TTA has no variance left to average.
#  (Measured on the aug-on runs: weighted consistency ~0.0002, flip-TTA +0.0006,
#   4 extra D4 rotation views +0.0002.)
#
#  ARMS
#    A       aug OFF, FCT off              -> what orientation aug is worth
#    A-TTA   same ckpt, 4-view flip TTA    -> TTA headroom with NO invariance training
#    B       aug OFF, FCT consistency-only -> can consistency REPLACE augmentation?
#    C       same ckpt as B, 4-view TTA    -> does TTA finally have headroom?
#    D       aug ON, FCT, TTA              -> already trained, nothing to run here
#
#  READ: C > D confirms the redundancy thesis. B >= A means consistency is a
#  viable substitute for augmentation. A << B means augmentation was carrying
#  the invariance all along and consistency alone cannot replace it -- still a
#  publishable result ("substitutes, not complements"), just the other sign.
#
#  GPU 0 runs arm B (the long one, ~3 forwards/step). GPU 1 runs arm A and
#  finishes early, so it also picks up its own two inference passes.
#  Wall-clock is bounded by GPU 0.
#
#  Usage:  bash run_ablation_parallel.sh
#  Logs:   logs/ablation/*.log
# =============================================================================
set -u

DT_PATH="/home/yassine/projects/Polyp-PVT/result_map/PolypPVT"
LOGDIR="logs/ablation"
mkdir -p "$LOGDIR" results_cl

CFG_A="configs/UACANet_refine_polyppvt_noaug.yaml"
CFG_B="configs/UACANet_refine_polyppvt_noaug_fct.yaml"
CKPT_A="checkpoints/UACANet_refine_polyppvt_noaug/best.pth"
CKPT_B="checkpoints/UACANet_refine_polyppvt_noaug_fct/best.pth"

for f in "$CFG_A" "$CFG_B"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# --- guard: the whole ablation is void if these two slip ---------------------
python - <<'PY' || exit 1
import sys, yaml
b = yaml.safe_load(open('configs/UACANet_refine_polyppvt_noaug_fct.yaml'))
a = yaml.safe_load(open('configs/UACANet_refine_polyppvt_noaug.yaml'))
ok = True
if b['Model'].get('fct_supervise_flips') is not False:
    print("FATAL: arm B has fct_supervise_flips != False -> collapses into arm D"); ok = False
for name, cfg in (('A', a), ('B', b)):
    tl = cfg['Train']['Dataset']['transform_list']
    leaked = [k for k in ('random_flip', 'random_rotate') if k in tl]
    if leaked:
        print(f"FATAL: arm {name} still has orientation aug: {leaked}"); ok = False
if ok:
    print("pre-flight OK: orientation aug removed from A and B; B is consistency-only")
sys.exit(0 if ok else 1)
PY

# =============================================================================
#  GPU 0  --  arm B (train)  ->  arm B (no-TTA)  ->  arm C (TTA)
# =============================================================================
gpu0() {
  echo "[gpu0] $(date +%T) train arm B (aug OFF + FCT consistency-only)"
  CUDA_VISIBLE_DEVICES=0 python run/Train_patch.py --config "$CFG_B" --verbose \
      > "$LOGDIR/B_train.log" 2>&1 || { echo "[gpu0] arm B TRAIN FAILED"; return 1; }

  echo "[gpu0] $(date +%T) score arm B (no TTA)"
  CUDA_VISIBLE_DEVICES=0 python run/Test_patches.py --config "$CFG_B" \
      --pth "$CKPT_B" --dt_path "$DT_PATH" \
      --out_dir results_cl/ABL_B_noaug_fct_noTTA \
      > "$LOGDIR/B_test.log" 2>&1 || echo "[gpu0] arm B TEST FAILED"

  echo "[gpu0] $(date +%T) score arm C (4-view TTA)"
  CUDA_VISIBLE_DEVICES=0 python run/Test_patch_tta.py --config "$CFG_B" \
      --pth "$CKPT_B" --dt_path "$DT_PATH" \
      --out_dir results_cl/ABL_C_noaug_fct_TTA \
      > "$LOGDIR/C_test.log" 2>&1 || echo "[gpu0] arm C TEST FAILED"

  echo "[gpu0] $(date +%T) done"
}

# =============================================================================
#  GPU 1  --  arm A (train)  ->  arm A (no-TTA)  ->  arm A (TTA, free control)
# =============================================================================
gpu1() {
  echo "[gpu1] $(date +%T) train arm A (aug OFF, no FCT)"
  CUDA_VISIBLE_DEVICES=1 python run/Train_patch.py --config "$CFG_A" --verbose \
      > "$LOGDIR/A_train.log" 2>&1 || { echo "[gpu1] arm A TRAIN FAILED"; return 1; }

  echo "[gpu1] $(date +%T) score arm A (no TTA)"
  CUDA_VISIBLE_DEVICES=1 python run/Test_patches.py --config "$CFG_A" \
      --pth "$CKPT_A" --dt_path "$DT_PATH" \
      --out_dir results_cl/ABL_A_noaug_noTTA \
      > "$LOGDIR/A_test.log" 2>&1 || echo "[gpu1] arm A TEST FAILED"

  # Free and highly informative: TTA on a model with NO invariance training at
  # all. This is the upper bound on how much variance flip-TTA can ever exploit.
  echo "[gpu1] $(date +%T) score arm A + TTA (control)"
  CUDA_VISIBLE_DEVICES=1 python run/Test_patch_tta.py --config "$CFG_A" \
      --pth "$CKPT_A" --dt_path "$DT_PATH" \
      --out_dir results_cl/ABL_A_noaug_TTA \
      > "$LOGDIR/A_tta_test.log" 2>&1 || echo "[gpu1] arm A+TTA TEST FAILED"

  echo "[gpu1] $(date +%T) done"
}

gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; R0=$?
wait $P1; R1=$?

echo
echo "================ ablation finished (gpu0=$R0 gpu1=$R1) ================"
echo "score these four folders against your GT and compare to the aug-on arms:"
echo "  results_cl/ABL_A_noaug_noTTA        <- arm A"
echo "  results_cl/ABL_A_noaug_TTA          <- arm A + TTA (control)"
echo "  results_cl/ABL_B_noaug_fct_noTTA    <- arm B"
echo "  results_cl/ABL_C_noaug_fct_TTA      <- arm C   *** the one that matters"
echo "logs in $LOGDIR"
