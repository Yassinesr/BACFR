#!/usr/bin/env bash
# =============================================================================
#  Re-score the no-aug arms after the roi_align aligned=True fix (022b4ecb7d)
# =============================================================================
#  Training patches were cropped with mmcv (aligned=True); the test scripts
#  cropped with torchvision (aligned=False) -> a 2x2 blur plus a half-pixel
#  shift on every test patch. Every refined number was produced with that
#  mismatch, so these four arms need re-scoring.
#
#  ARMS
#    A       aug OFF, no FCT, no TTA
#    A+TTA   same checkpoint, 4-view flip TTA
#    B       aug OFF, FCT consistency-only
#    C       same checkpoint as B, 4-view flip TTA
#
#  NOT re-run, and unaffected: raw Polyp-PVT (0.8683). No refinement, no
#  roi_align, so that baseline still stands as the reference point.
#
#  Inference only - no retraining, checkpoints unchanged.
#  Usage: bash rerun_aligned.sh
# =============================================================================
set -u
DT="/home/yassine/projects/Polyp-PVT/result_map/PolypPVT"
CFG_A="configs/UACANet_refine_polyppvt_noaug.yaml"
CFG_B="configs/UACANet_refine_polyppvt_noaug_fct.yaml"
CK_A="checkpoints/UACANet_refine_polyppvt_noaug"
CK_B="checkpoints/UACANet_refine_polyppvt_noaug_fct"
mkdir -p logs/aligned results_cl

# run <gpu> <config> <ckpt_dir> <tta:0|1> <out_name>
run() {
  local gpu=$1 cfg=$2 ck=$3 tta=$4 out=$5
  [ -f "$cfg" ] || { echo "[skip] no config $cfg"; return 0; }
  local pth="$ck/best.pth"
  if [ ! -f "$pth" ]; then
    pth=$(ls -1v "$ck"/epoch_*.pth 2>/dev/null | tail -1)
    [ -z "$pth" ] && { echo "[skip] no checkpoint in $ck"; return 0; }
    echo "[note] $out: best.pth missing, using $(basename "$pth")"
  fi
  local script=run/Test_patches.py; [ "$tta" = 1 ] && script=run/Test_patch_tta.py
  echo "[gpu$gpu] $(date +%T) $out"
  CUDA_VISIBLE_DEVICES=$gpu python $script --config "$cfg" --pth "$pth" \
      --dt_path "$DT" --out_dir "results_cl/$out" > "logs/aligned/$out.log" 2>&1 \
      || echo "[gpu$gpu] FAILED: $out (see logs/aligned/$out.log)"
}

# GPU 0 takes the FCT checkpoint, GPU 1 the no-FCT one: independent, balanced.
( run 0 "$CFG_B" "$CK_B" 0 ALIGNED_B_noaug_fct_noTTA
  run 0 "$CFG_B" "$CK_B" 1 ALIGNED_C_noaug_fct_TTA
  echo "[gpu0] $(date +%T) done" ) & P0=$!
( run 1 "$CFG_A" "$CK_A" 0 ALIGNED_A_noaug_noTTA
  run 1 "$CFG_A" "$CK_A" 1 ALIGNED_A_noaug_TTA
  echo "[gpu1] $(date +%T) done" ) & P1=$!
wait $P0; wait $P1

echo; echo "=== done - score these four, compare to the pre-fix numbers ==="
printf '  %-32s pre-fix mean Dice\n' ""
printf '  %-32s %s\n' results_cl/ALIGNED_A_noaug_noTTA     0.8717
printf '  %-32s %s\n' results_cl/ALIGNED_A_noaug_TTA       0.8720
printf '  %-32s %s\n' results_cl/ALIGNED_B_noaug_fct_noTTA 0.8723
printf '  %-32s %s\n' results_cl/ALIGNED_C_noaug_fct_TTA   0.8723
echo "  CVC-300 is the blocker: pre-fix 0.9037 / 0.9046 / 0.9032 / 0.9032, need 0.9101 vs UACANet-L"
