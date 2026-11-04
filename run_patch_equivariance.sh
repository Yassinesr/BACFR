#!/usr/bin/env bash
# =============================================================================
#  Patch-refiner equivariance gap -- the measurement that closes the argument
# =============================================================================
#  On WHOLE IMAGES, a single flipped view costs the Polyp-PVT backbone 0.0230
#  mean Dice (h -0.0135, v -0.0316, hv -0.0238), traced to its stem sampling on
#  a grid flush to one edge. That gap is what flip-TTA averages over, and why
#  averaging misregistered views costs -0.0355 there.
#
#  The patch refiner should show almost NO gap: its output is the coarse mask
#  plus learned residuals, and the anchor does not care which way round the
#  patch went in. If the gap here is ~0 while the whole-image gap is -0.023,
#  that pair of numbers IS the explanation for every null result on the patch
#  side -- no variance for TTA to average, no non-equivariance for FCT to fix.
#
#  Until now that claim was inferred from two facts (removing orientation
#  augmentation cost nothing, TTA gained nothing) rather than measured.
#
#  Usage: bash run_patch_equivariance.sh [checkpoint_dir]
# =============================================================================
set -u
CK="${1:-checkpoints/BACFR_Enhanced_v3_3_mixed}"
CFG="${CFG:-configs/BACFR_Enhanced_v3_3_mixed.yaml}"
DT="${DT:-/home/yassine/projects/Polyp-PVT/result_map/PolypPVT}"
PTH="$CK/best.pth"
[ -f "$PTH" ] || PTH=$(ls -1v "$CK"/epoch_*.pth 2>/dev/null | tail -1)
[ -z "${PTH:-}" ] && { echo "no checkpoint in $CK"; exit 1; }
echo "checkpoint: $PTH"

for v in plain h v hv; do
  echo "---- single_view=$v"
  python run/Test_patch_tta.py --config "$CFG" --pth "$PTH" --dt_path "$DT" \
      --single_view "$v" --out_dir "results_cl/EQV_patch_$v" \
      || echo "[FAILED] $v"
done

cat <<'NOTE'

Score the four folders. `plain` is the reference; h/v/hv are the gaps.

  results_cl/EQV_patch_plain   no flip (reference)
  results_cl/EQV_patch_h       horizontal
  results_cl/EQV_patch_v       vertical
  results_cl/EQV_patch_hv      both

Whole-image gaps to compare against:  h -0.0135   v -0.0316   hv -0.0238
NOTE
