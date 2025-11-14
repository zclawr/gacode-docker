#!/usr/bin/env bash
set -euo pipefail

# === Load .env ===
if [ -f .env ]; then
  echo "🔧 Loading environment from .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "❌ .env file not found. Aborting."
  exit 1
fi

# Required env in .env:
#   S3_BUCKET_NAME=ai-fusion-ga
#   S3_ENDPOINT_URL=https://s3-west.nrp-nautilus.io
# Optional:
#   AWS_CA_BUNDLE=/path/to/nautilus-ca.pem   # preferred over --insecure
#   S5_INSECURE=true                         # set if no CA bundle available

# === Validate input ===
if [[ -z "${1:-}" || -z "${2:-}" ]]; then
  echo "Usage: $0 [tglf|cgyro] path/to/local/dir"
  exit 1
fi

RUN_SIM_TYPE="$1"          # keep the requested run type separate
LOCAL_INPUT_DIR="$2"

if [[ "$RUN_SIM_TYPE" != "tglf" && "$RUN_SIM_TYPE" != "cgyro" ]]; then
  echo "❌ SIM_TYPE must be 'tglf' or 'cgyro'"
  exit 1
fi

if [[ ! -d "$LOCAL_INPUT_DIR" ]]; then
  echo "❌ Provided path $LOCAL_INPUT_DIR is not a directory."
  exit 1
fi

# === Ensure s5cmd present ===
if ! command -v s5cmd >/dev/null 2>&1; then
  echo "❌ s5cmd not found. Install with: brew install s5cmd"
  exit 1
fi

# === S3/Ceph safety knobs ===
export AWS_S3_FORCE_PATH_STYLE=true
export AWS_EC2_METADATA_DISABLED=true
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}
export AWS_REGION=${AWS_REGION:-us-west-2}

# Avoid proxies that can mangle streams/chunked requests
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy || true

# Build a reusable s5cmd flag list (endpoint, TLS settings, concurrency, retries)
S5_FLAGS=(--endpoint-url "${S3_ENDPOINT_URL}" --stat)
if [[ -n "${AWS_CA_BUNDLE:-}" && -f "${AWS_CA_BUNDLE}" ]]; then
  # prefer proper CA bundle
  export AWS_CA_BUNDLE
else
  # fall back to --insecure if requested
  if [[ "${S5_INSECURE:-}" == "true" ]]; then
    S5_FLAGS+=(--insecure)
  fi
fi

# === Setup run folder and S3 path ===
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/${DATE_TAG}"
mkdir -p "$RUN_DIR"
S3_BASE="cgyro-inputs-wesley-rho-sep/${DATE_TAG}/"

echo "📤 Uploading inputs from $LOCAL_INPUT_DIR to s3://${S3_BUCKET_NAME}/${S3_BASE}"

# Keep separate lists so we can choose which to run later
S3PATH_LIST_TGLF=()
S3PATH_LIST_CGYRO=()

# Helper to s5cmd sync (with trailing slashes preserved)
s5_sync_dir() {
  local src="$1"
  local dst="$2"
  # Ensure trailing slashes so s5cmd sync preserves relative structure
  [[ "${src}" != */ ]] && src="${src}/"
  [[ "${dst}" != */ ]] && dst="${dst}/"
  echo "📦 s5cmd sync: ${src} -> ${dst}"
  s5cmd "${S5_FLAGS[@]}" sync "${src}" "${dst}"
}

# === Sync both sim types if present ===
shopt -s nullglob
for batch_dir in "$LOCAL_INPUT_DIR"/batch-*; do
  [[ -d "$batch_dir" ]] || continue

  for sim in tglf cgyro; do
    sim_dir="$batch_dir/$sim"
    [[ -d "$sim_dir" ]] || continue

    # validate presence of the right input files
    valid_input_found=false
    for input_dir in "$sim_dir"/input-*; do
      if [[ "$sim" == "cgyro" && -f "$input_dir/input.cgyro" ]] || \
         [[ "$sim" == "tglf"  && -f "$input_dir/input.tglf"  ]]; then
        valid_input_found=true
        break
      fi
    done

    if $valid_input_found; then
      # Build REL_PATH relative to LOCAL_INPUT_DIR (without leading slash)
      REL_PATH="${sim_dir#$LOCAL_INPUT_DIR/}"

      # Destination s3 URL
      S3_DST="s3://${S3_BUCKET_NAME}/${S3_BASE}${REL_PATH}/"

      # Perform sync via s5cmd
      s5_sync_dir "$sim_dir" "$S3_DST"

      # Append to selection lists (quote the path for YAML array)
      if [[ "$sim" == "tglf" ]]; then
        S3PATH_LIST_TGLF+=("\"${S3_BASE}${REL_PATH}/\"")
      else
        S3PATH_LIST_CGYRO+=("\"${S3_BASE}${REL_PATH}/\"")
      fi
    else
      echo "⏩ Skipping $sim_dir (no valid input files)"
    fi
  done
done
shopt -u nullglob

# === Choose the paths for the requested run type ===
if [[ "$RUN_SIM_TYPE" == "tglf" ]]; then
  SELECTED_PATHS=("${S3PATH_LIST_TGLF[@]}")
else
  SELECTED_PATHS=("${S3PATH_LIST_CGYRO[@]}")
fi

if [[ ${#SELECTED_PATHS[@]} -eq 0 ]]; then
  echo "❌ No synced paths found for run type '$RUN_SIM_TYPE'. Nothing to run."
  exit 1
fi

JOINED_PATHS=$(IFS=, ; echo "${SELECTED_PATHS[*]}")

# === Create launch.yaml and metadata ===
YAML_ARCHIVE="$RUN_DIR/launch_${RUN_SIM_TYPE}.yaml"
YAML_CONFIG="./config/launch.yaml"

for TARGET_YAML in "$YAML_ARCHIVE" "$YAML_CONFIG"; do
  cp ./config/launch_template.yaml "$TARGET_YAML"
  cat >> "$TARGET_YAML" <<EOF
dataset:
  default:
    hparam:
      _s3path: [${JOINED_PATHS}]
run:
  model: [${RUN_SIM_TYPE}]
  dataset: [default]
EOF
done

# Save the full lists for reference
printf "%s\n" "${S3PATH_LIST_TGLF[@]}"   > "$RUN_DIR/s3paths_tglf.txt"
printf "%s\n" "${S3PATH_LIST_CGYRO[@]}"  > "$RUN_DIR/s3paths_cgyro.txt"
printf "%s\n" "${SELECTED_PATHS[@]}"     > "$RUN_DIR/s3paths_${RUN_SIM_TYPE}.txt"

echo "✅ Saved launch YAML to:"
echo "   - $YAML_ARCHIVE"
echo "   - $YAML_CONFIG"

# === Kick off jobs for the requested type only ===
make job
