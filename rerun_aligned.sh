#!/usr/bin/env bash
# =============================================================================
#  Re-score every refined arm after the roi_align aligned=True fix (022b4ecb7d)
# =============================================================================
#  Training patches were cropped with mmcv (aligned=True); the test scripts
#  cropped with torchvision (aligned=False) -> a 2x2 blur plus a half-pixel
#  shift on every test patch. Every refined number ever reported on this repo
#  was produced with that mismatch, so all of them need re-scoring.
#
#  NOT affected, do NOT re-run: raw Polyp-PVT (0.8683). No refinement, no
#  roi_align. That baseline still stands.
#
#  Inference only - no retraining, checkpoints are unchanged.
#  Usage: bash rerun_aligned.sh          (runs everything, 2 GPUs)
#         bash rerun_aligned.sh 1        (tier 1 only - the reported headline)
# =============================================================================
set -u
DT="/home/yassine/projects/Polyp-PVT/result_map/PolypPVT"
TIER="${1:-9}"
mkdir -p logs/aligned results_cl

# run <gpu> <tier> <config> <ckpt_dir> <tta:0|1> <out_name>
run() {
  local gpu=$1 tier=$2 cfg=$3 ck=$4 tta=$5 out=$6
  [ "$tier" -gt "$TIER" ] && return 0
  local pth="$ck/best.pth"
  if [ ! -f "$cfg" ]; then echo "[skip] no config $cfg"; return 0; fi
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

gpu0() {
  # --- tier 1: the three headline BACFR arms (reported with TTA) -------------
  run 0 1 configs/BACFR_Enhanced_v3_3_mixed.yaml    checkpoints/BACFR_Enhanced_v3_3_mixed        1 ALIGNED_mixed_TTA
  run 0 1 configs/BACFR_Enhanced_v3_3.yaml          checkpoints/BACFR_Enhanced_v3_flipconsistency 1 ALIGNED_pranet_TTA
  # --- tier 2: the no-aug ablation, FCT half --------------------------------
  run 0 2 configs/UACANet_refine_polyppvt_noaug_fct.yaml checkpoints/UACANet_refine_polyppvt_noaug_fct 0 ALIGNED_B_noaug_fct_noTTA
  run 0 2 configs/UACANet_refine_polyppvt_noaug_fct.yaml checkpoints/UACANet_refine_polyppvt_noaug_fct 1 ALIGNED_C_noaug_fct_TTA
  # --- tier 3: UACANet_Refine baseline arms ---------------------------------
  run 0 3 configs/UACANet_refine_polyppvt.yaml      checkpoints/UACANet_refine_polyppvt          0 ALIGNED_refine_noTTA
  run 0 3 configs/UACANet_refine_polyppvt.yaml      checkpoints/UACANet_refine_polyppvt          1 ALIGNED_refine_TTA
  echo "[gpu0] $(date +%T) done"
}

gpu1() {
  run 1 1 configs/BACFR_Enhanced_v3_3_polyppvt.yaml checkpoints/BACFR_Enhanced_v3_3_polyppvt     1 ALIGNED_polyppvt_TTA
  run 1 2 configs/UACANet_refine_polyppvt_noaug.yaml checkpoints/UACANet_refine_polyppvt_noaug   0 ALIGNED_A_noaug_noTTA
  run 1 2 configs/UACANet_refine_polyppvt_noaug.yaml checkpoints/UACANet_refine_polyppvt_noaug   1 ALIGNED_A_noaug_TTA
  run 1 3 configs/UACANet_refine_polyppvt_fct.yaml  checkpoints/UACANet_refine_polyppvt_fct      0 ALIGNED_refine_fct_noTTA
  run 1 3 configs/UACANet_refine_polyppvt_fct.yaml  checkpoints/UACANet_refine_polyppvt_fct      1 ALIGNED_refine_fct_TTA
  run 1 3 configs/UACANet_refine_polyppvt_fct_conly.yaml checkpoints/UACANet_refine_polyppvt_fct_conly 1 ALIGNED_refine_conly_TTA
  echo "[gpu1] $(date +%T) done"
}

gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; wait $P1
echo; echo "=== done. score every results_cl/ALIGNED_* folder ==="
ls -d results_cl/ALIGNED_* 2>/dev/null
