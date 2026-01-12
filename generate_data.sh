#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Assumes:
#   conda activate pyro
#   python resolves to the pyro env
# --------------------------------------------------

# ---------------- CONFIG ----------------
SRC_H5="combined_9k_minmax_norm_rho_all_new_dec.h5"

CONVERTER="input_generation/h5_to_cgyro_input.py"

OUT_100="./cgyro_inputs"
OUT_50="./cgyro_inputs_fast"

BATCH_SIZE_100=25
BATCH_SIZE_50=25

# Leave empty to auto-pick cpu_count()-1
WORKERS=""

# ---------------- SEED GENERATION ----------------
SEED_100=$(( (RANDOM << 15) | RANDOM ))
SEED_50=$(( (RANDOM << 15) | RANDOM ))

DST_H5_100="combined_9k_minmax_norm_new_rho_all_sample100_seed${SEED_100}.h5"
DST_H5_50="combined_9k_minmax_norm_new_rho_all_sample50_seed${SEED_50}.h5"

# ---------------- HELPERS ----------------
run_convert () {
  local h5="$1"
  local out="$2"
  local batch_size="$3"

  mkdir -p "$out"

  if [[ -n "${WORKERS}" ]]; then
    python "$CONVERTER" \
      --h5 "$h5" \
      --out "$out" \
      --batch-size "$batch_size" \
      --workers "$WORKERS"
  else
    python "$CONVERTER" \
      --h5 "$h5" \
      --out "$out" \
      --batch-size "$batch_size"
  fi
}

# ---------------- RUN ----------------
echo "==> Sampling k=100  seed=${SEED_100}"
python input_generation/sample_h5.py \
  --src "$SRC_H5" \
  --dst "$DST_H5_100" \
  --k 100 \
  --seed "$SEED_100"

echo "==> Sampling k=50   seed=${SEED_50}"
python input_generation/sample_h5.py \
  --src "$SRC_H5" \
  --dst "$DST_H5_50" \
  --k 50 \
  --seed "$SEED_50"

echo "==> Converting 100-sample H5 -> ${OUT_100}"
run_convert "$DST_H5_100" "$OUT_100" "$BATCH_SIZE_100"

echo "==> Converting 50-sample H5 -> ${OUT_50}"
run_convert "$DST_H5_50" "$OUT_50" "$BATCH_SIZE_50"

echo "✅ Done."
echo "   100-sample seed: ${SEED_100}"
echo "   50-sample seed:  ${SEED_50}"
echo "   100-sample file: ${DST_H5_100}"
echo "   50-sample file:  ${DST_H5_50}"
