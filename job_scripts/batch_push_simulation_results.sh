#!/usr/bin/env bash
set -euo pipefail

#------------------------------------------------------------------
# batch_push_simulation_results.sh
#
# PURPOSE:
#  Push *already completed* simulation runs to S3, using the same
#  bucket, prefix and key layout that batch_run_simulation_jobs.sh
#  establishes for inputs and that the k8s job (config/launch.yaml)
#  establishes for results.
#
#  Use this when the runs were executed somewhere other than the
#  Nautilus jobs -- e.g. submitted at NERSC with
#  job_scripts/submit_cgyro_nersc.sh -- and only the upload half of
#  the workflow still has to happen.
#
# LAYOUT EXPECTED (same as ./cgyro_inputs):
#  <local-dir>/batch-XXX/{cgyro,tglf}/input-YYY/<inputs + outputs>
#
# WHAT GETS PUSHED:
#  A batch qualifies on its *cgyro* runs: >=1 completed cgyro run means the
#  batch is pushed.  TGLF is never executed, but its inputs are needed in
#  post-processing, so every tglf subdirectory of a qualifying batch is
#  pushed alongside the cgyro results, regardless of completion.
#
# S3 LAYOUT PRODUCED:
#  <prefix>/<DATE_TAG>/batch-XXX/<sim>/input-YYY/...      (raw sync)
#  <prefix>/<DATE_TAG>/batch-XXX/compressed_outputs.tar.xz (cgyro)
#
#  The tarball members are prefixed with the full s3 path, exactly as
#  `tar -cJvf <s3path>compressed_outputs.tar.xz <s3path>` produces
#  them inside the cgyro job container.
#
# USAGE:
#  ./job_scripts/batch_push_simulation_results.sh [tglf|cgyro] path/to/dir [options]
#  For example:
#  ./job_scripts/batch_push_simulation_results.sh cgyro ./cgyro_inputs
#  ./job_scripts/batch_push_simulation_results.sh cgyro ./cgyro_inputs -t 20260910_130450
#------------------------------------------------------------------

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

usage () {
  cat <<'EOF'

Usage:   batch_push_simulation_results.sh [tglf|cgyro] path/to/local/dir [options]

         -t, --date-tag <tag>
         Timestamp tag to publish under, e.g. 20260910_130450.  Pass the
         tag of the original input upload to place results next to their
         inputs.  [default: current time]

         --prefix <prefix>
         S3 prefix above the date tag.
         [default: cgyro-inputs-wesley-rho-sep]

         --no-sync
         Do not sync the raw run directories; only upload the tarball.

         --no-tar
         Do not build/upload compressed_outputs.tar.xz (cgyro only).

         --all
         Upload every batch, including those with no completed cgyro runs.
         [default: batches with no completed cgyro run are skipped]

         --dry-run
         Print what would be transferred without writing to S3.

EOF
  exit 1
}

# === Validate input ===
if [[ -z "${1:-}" || -z "${2:-}" ]]; then
  usage
fi

RUN_SIM_TYPE="$1"          # keep the requested run type separate
LOCAL_INPUT_DIR="$2"
shift 2

DATE_TAG=""
S3_PREFIX="cgyro-inputs-wesley-rho-sep"
DO_SYNC=1
DO_TAR=1
REQUIRE_COMPLETE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--date-tag) shift ; DATE_TAG="${1:-}" ;;
    --prefix)      shift ; S3_PREFIX="${1:-}" ;;
    --no-sync)     DO_SYNC=0 ;;
    --no-tar)      DO_TAR=0 ;;
    --all)         REQUIRE_COMPLETE=0 ;;
    --dry-run)     DRY_RUN=1 ;;
    -h|-help|--help) usage ;;
    *) echo "❌ Unknown option: $1" ; usage ;;
  esac
  shift
done

if [[ "$RUN_SIM_TYPE" != "tglf" && "$RUN_SIM_TYPE" != "cgyro" ]]; then
  echo "❌ SIM_TYPE must be 'tglf' or 'cgyro'"
  exit 1
fi

if [[ ! -d "$LOCAL_INPUT_DIR" ]]; then
  echo "❌ Provided path $LOCAL_INPUT_DIR is not a directory."
  exit 1
fi

if [[ $DO_SYNC -eq 0 && $DO_TAR -eq 0 ]]; then
  echo "❌ --no-sync and --no-tar leave nothing to upload."
  exit 1
fi

# === Ensure s5cmd present ===
if ! command -v s5cmd >/dev/null 2>&1; then
  echo "❌ s5cmd not found. Install with: brew install s5cmd"
  exit 1
fi

if [[ "$RUN_SIM_TYPE" == "cgyro" && $DO_TAR -eq 1 ]]; then
  if ! tar --help >/dev/null 2>&1; then
    echo "❌ tar not found."
    exit 1
  fi
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
if [[ $DRY_RUN -eq 1 ]]; then
  S5_FLAGS+=(--dry-run)
fi

# === Setup run folder and S3 path ===
if [[ -z "$DATE_TAG" ]]; then
  DATE_TAG=$(date +"%Y%m%d_%H%M%S")
fi
RUN_DIR="runs/${DATE_TAG}"
mkdir -p "$RUN_DIR"
S3_BASE="${S3_PREFIX%/}/${DATE_TAG}/"

echo "📤 Uploading completed $RUN_SIM_TYPE runs from $LOCAL_INPUT_DIR to s3://${S3_BUCKET_NAME}/${S3_BASE}"
[[ $DRY_RUN -eq 1 ]] && echo "🧪 Dry run: nothing will be written to S3."

# Keep separate lists so the paths can be referenced later
S3PATH_LIST_TGLF=()
S3PATH_LIST_CGYRO=()

# Staging tree used to give tar members the full s3-path prefix without
# copying any data (symlink + tar -h).
TAR_STAGE=""
cleanup () {
  [[ -n "$TAR_STAGE" && -d "$TAR_STAGE" ]] && rm -rf "$TAR_STAGE"
}
trap cleanup EXIT

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

# Helper to s5cmd cp a single file
s5_cp_file() {
  local src="$1"
  local dst="$2"
  echo "📦 s5cmd cp: ${src} -> ${dst}"
  s5cmd "${S5_FLAGS[@]}" cp "${src}" "${dst}"
}

# Count completed runs in a sim dir (cgyro: out.cgyro.info, tglf: out.tglf.*)
count_completed () {
  local sim="$1"
  local sim_dir="$2"
  local n=0
  local input_dir
  for input_dir in "$sim_dir"/input-*; do
    [[ -d "$input_dir" ]] || continue
    if [[ "$sim" == "cgyro" ]]; then
      [[ -f "$input_dir/out.cgyro.info" ]] && n=$((n + 1))
    else
      compgen -G "$input_dir/out.tglf.*" >/dev/null && n=$((n + 1))
    fi
  done
  echo "$n"
}

# Build <prefix>/<date>/batch-XXX/compressed_outputs.tar.xz for one batch
# and upload it, mirroring the cgyro job container's tar invocation.
tar_and_upload_batch () {
  local batch_dir="$1"
  local batch_rel="$2"                     # e.g. batch-000
  local member="${S3_BASE}${batch_rel}"    # e.g. <prefix>/<date>/batch-000
  local abs_batch
  abs_batch=$(cd "$batch_dir" && pwd)

  if [[ -z "$TAR_STAGE" ]]; then
    TAR_STAGE=$(mktemp -d "${TMPDIR:-/tmp}/gacode-results.XXXXXX")
  fi

  local stage_parent="${TAR_STAGE}/${S3_BASE%/}"
  mkdir -p "$stage_parent"
  rm -f "$stage_parent/$batch_rel"
  ln -s "$abs_batch" "$stage_parent/$batch_rel"

  local tarball="${TAR_STAGE}/${batch_rel}-compressed_outputs.tar.xz"
  echo "🗜  tar -cJf ${member}/ -> ${batch_rel}/compressed_outputs.tar.xz"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "🧪 [dry run] would archive ${abs_batch} as ${member}/"
  else
    # -h follows the staging symlink so the archive holds real files
    tar -C "$TAR_STAGE" -chJf "$tarball" "$member"
    s5_cp_file "$tarball" "s3://${S3_BUCKET_NAME}/${S3_BASE}${batch_rel}/compressed_outputs.tar.xz"
    rm -f "$tarball"
  fi
  rm -f "$stage_parent/$batch_rel"
}

# === Push qualifying batches (gate on cgyro, always carry tglf along) ===
n_pushed=0
n_skipped=0
n_partial=0

shopt -s nullglob
for batch_dir in "$LOCAL_INPUT_DIR"/batch-*; do
  [[ -d "$batch_dir" ]] || continue

  # batch-relative path, for the YAML/manifest outer-dir paths
  BATCH_REL="${batch_dir#$LOCAL_INPUT_DIR/}"        # e.g. "batch-000"
  YAML_BATCH_PATH="${S3_BASE}${BATCH_REL}/"         # e.g. "cgyro-inputs.../DATE/batch-000/"

  # Only cgyro decides whether this batch is worth pushing -- tglf is never
  # run, so it can never make a batch eligible on its own.
  cgyro_dir="$batch_dir/cgyro"
  n_total=0
  n_done=0
  if [[ -d "$cgyro_dir" ]]; then
    for input_dir in "$cgyro_dir"/input-*; do
      [[ -d "$input_dir" ]] && n_total=$((n_total + 1))
    done
    n_done=$(count_completed cgyro "$cgyro_dir")
  fi

  if [[ $n_done -eq 0 && $REQUIRE_COMPLETE -eq 1 ]]; then
    echo "⏩ Skipping $batch_dir (0/$n_total completed cgyro runs; use --all to push anyway)"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  if [[ $n_done -lt $n_total ]]; then
    echo "⚠️  $batch_dir: only $n_done/$n_total cgyro runs completed"
    n_partial=$((n_partial + 1))
  else
    echo "✅ $batch_dir: $n_done/$n_total cgyro runs completed"
  fi

  # cgyro results plus *every* tglf subdirectory of this batch, since
  # post-processing needs the tglf inputs next to the cgyro outputs.
  for sim_dir in "$cgyro_dir" "$batch_dir"/tglf*; do
    [[ -d "$sim_dir" ]] || continue
    sim="$(basename "$sim_dir")"

    if [[ "$sim" != "cgyro" ]]; then
      n_tglf=0
      for input_dir in "$sim_dir"/input-*; do
        [[ -d "$input_dir" ]] && n_tglf=$((n_tglf + 1))
      done
      echo "➕ $sim_dir: carrying $n_tglf tglf input dir(s) along (not run)"
    fi

    if [[ $DO_SYNC -eq 1 ]]; then
      # REL_PATH relative to LOCAL_INPUT_DIR, e.g. "batch-000/cgyro"
      REL_PATH="${sim_dir#$LOCAL_INPUT_DIR/}"
      S3_DST="s3://${S3_BUCKET_NAME}/${S3_BASE}${REL_PATH}/"
      s5_sync_dir "$sim_dir" "$S3_DST"
    fi

    # Manifests reference the *outer* batch dir, as the launch YAML does
    if [[ "$sim" == "cgyro" ]]; then
      S3PATH_LIST_CGYRO+=("\"${YAML_BATCH_PATH}\"")
    else
      S3PATH_LIST_TGLF+=("\"${YAML_BATCH_PATH}\"")
    fi
  done

  # The cgyro job packages the whole batch dir (cgyro + tglf) as one archive.
  if [[ "$RUN_SIM_TYPE" == "cgyro" && $DO_TAR -eq 1 ]]; then
    tar_and_upload_batch "$batch_dir" "$BATCH_REL"
  fi

  n_pushed=$((n_pushed + 1))
done
shopt -u nullglob

if [[ $n_pushed -eq 0 ]]; then
  echo "❌ No completed cgyro runs found under $LOCAL_INPUT_DIR. Nothing uploaded."
  exit 1
fi

# === Choose the paths for the requested run type ===
if [[ "$RUN_SIM_TYPE" == "tglf" ]]; then
  SELECTED_PATHS=("${S3PATH_LIST_TGLF[@]}")
else
  SELECTED_PATHS=("${S3PATH_LIST_CGYRO[@]}")
fi

if [[ ${#SELECTED_PATHS[@]} -eq 0 ]]; then
  echo "❌ No pushed paths found for run type '$RUN_SIM_TYPE'."
  exit 1
fi

# Save the full lists for reference (same names batch_run_simulation_jobs.sh uses)
if [[ ${#S3PATH_LIST_TGLF[@]} -gt 0 ]]; then
  printf "%s\n" "${S3PATH_LIST_TGLF[@]}"  > "$RUN_DIR/s3paths_tglf.txt"
fi
if [[ ${#S3PATH_LIST_CGYRO[@]} -gt 0 ]]; then
  printf "%s\n" "${S3PATH_LIST_CGYRO[@]}" > "$RUN_DIR/s3paths_cgyro.txt"
fi
printf "%s\n" "${SELECTED_PATHS[@]}"      > "$RUN_DIR/s3paths_${RUN_SIM_TYPE}.txt"

# Record where these results came from, since no launch YAML is generated
cat > "$RUN_DIR/results_${RUN_SIM_TYPE}.txt" <<EOF
source_dir: $(cd "$LOCAL_INPUT_DIR" && pwd)
s3_base:    s3://${S3_BUCKET_NAME}/${S3_BASE}
sim_type:   ${RUN_SIM_TYPE}
date_tag:   ${DATE_TAG}
batches:    ${n_pushed} pushed, ${n_skipped} skipped, ${n_partial} partial
pushed_at:  $(date +"%Y-%m-%d %H:%M:%S %Z")
EOF

echo
echo "✅ Pushed $n_pushed batch(es) ($n_skipped skipped, $n_partial partial) to:"
echo "   - s3://${S3_BUCKET_NAME}/${S3_BASE}"
echo "✅ Saved manifests to:"
echo "   - $RUN_DIR/s3paths_${RUN_SIM_TYPE}.txt"
echo "   - $RUN_DIR/results_${RUN_SIM_TYPE}.txt"
