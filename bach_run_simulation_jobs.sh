#!/bin/bash

# === Load .env ===
if [ -f .env ]; then
  echo "🔧 Loading environment from .env"
  set -a
  source .env
  set +a
else
  echo "❌ .env file not found. Aborting."
  exit 1
fi

# === Validate input ===
if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 [tglf|cgyro] path/to/local/dir"
  exit 1
fi

SIM_TYPE=$1
LOCAL_INPUT_DIR=$2

if [[ "$SIM_TYPE" != "tglf" && "$SIM_TYPE" != "cgyro" ]]; then
  echo "❌ SIM_TYPE must be 'tglf' or 'cgyro'"
  exit 1
fi

if [[ ! -d "$LOCAL_INPUT_DIR" ]]; then
  echo "❌ Provided path $LOCAL_INPUT_DIR is not a directory."
  exit 1
fi

# === Setup run folder and S3 path ===
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/${DATE_TAG}"
mkdir -p "$RUN_DIR"
S3_BASE="cgyro-inputs-wesley/${DATE_TAG}/"

echo "📤 Uploading $SIM_TYPE inputs from $LOCAL_INPUT_DIR to s3://${S3_BUCKET_NAME}/${S3_BASE}"
export AWS_S3_SIGNATURE_VERSION=s3v4

S3PATH_LIST=()

for batch_dir in "$LOCAL_INPUT_DIR"/batch-*; do
  sim_dir="$batch_dir/$SIM_TYPE"
  if [[ -d "$sim_dir" ]]; then
    valid_input_found=false
    for input_dir in "$sim_dir"/input-*; do
      if [[ "$SIM_TYPE" == "cgyro" && -f "$input_dir/input.cgyro" ]] || \
         [[ "$SIM_TYPE" == "tglf" && -f "$input_dir/input.tglf" ]]; then
        valid_input_found=true
        break
      fi
    done

    if $valid_input_found; then
      REL_PATH="${sim_dir#$LOCAL_INPUT_DIR/}"
      echo "📦 Syncing $REL_PATH..."
      aws s3 sync "$sim_dir" "s3://${S3_BUCKET_NAME}/${S3_BASE}${REL_PATH}/" \
        --exclude "*" --include "input-*/input.${SIM_TYPE}" \
        --endpoint-url "$S3_ENDPOINT_URL" --no-verify-ssl

      S3PATH_LIST+=("\"${S3_BASE}${REL_PATH}/\"")
    else
      echo "⏩ Skipping $sim_dir (no valid input files)"
    fi
  fi
done

# === Create launch.yaml and metadata ===
JOINED_PATHS=$(IFS=, ; echo "${S3PATH_LIST[*]}")
YAML_ARCHIVE="$RUN_DIR/launch_${SIM_TYPE}.yaml"
YAML_CONFIG="./config/launch.yaml"

for TARGET_YAML in "$YAML_ARCHIVE" "$YAML_CONFIG"; do
  cp ./config/launch_template.yaml "$TARGET_YAML"
  cat >> "$TARGET_YAML" <<EOF
dataset:
  default:
    hparam:
      _s3path: [${JOINED_PATHS}]
run:
  model: [${SIM_TYPE}]
  dataset: [default]
EOF
done



printf "%s\n" "${S3PATH_LIST[@]}" > "$RUN_DIR/s3paths_${SIM_TYPE}.txt"

echo "✅ Saved launch YAML to $YAML_PATH"

make job