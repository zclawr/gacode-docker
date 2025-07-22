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

if [[ ! -d "$LOCAL_INPUT_DIR" ]]; then
  echo "❌ Provided path $LOCAL_INPUT_DIR is not a directory."
  exit 1
fi

# === Setup base path in S3 ===
DATE_TAG=$(date +"%Y%m%d_%H%M%S")
S3_BASE="inputs/${DATE_TAG}/"

echo "📤 Uploading contents of $LOCAL_INPUT_DIR to s3://${S3_BUCKET_NAME}/${S3_BASE}"

# === Upload subdirs and collect full s3 paths ===
S3PATH_LIST=()

for subdir in "$LOCAL_INPUT_DIR"/*; do
  if [[ -d "$subdir" ]]; then
    name=$(basename "$subdir")
    echo "📦 Uploading $name..."
    aws s3 cp "$subdir" "s3://${S3_BUCKET_NAME}/${S3_BASE}${name}/" \
      --recursive --endpoint-url "$S3_ENDPOINT_URL"
    S3PATH_LIST+=("\"${S3_BASE}${name}/\"")
  fi
done

# === Join for YAML array ===
JOINED_PATHS=$(IFS=, ; echo "${S3PATH_LIST[*]}")

# === Generate launch.yaml ===
cp ./config/launch_template.yaml ./config/launch.yaml
cat >> ./config/launch.yaml <<EOF
dataset:
  default:
    hparam:
      _s3path: [${JOINED_PATHS}]
run:
  model: [${SIM_TYPE}]
  dataset: [default]
EOF

echo "✅ launch.yaml created with ${#S3PATH_LIST[@]} S3 paths"
make job
