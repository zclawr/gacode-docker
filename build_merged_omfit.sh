#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

# ----------------------------
# Config (override via env)
# ----------------------------
OMFIT_BRANCH="${OMFIT_BRANCH:-xarray-fix}"
OMFIT_REPO_URL="${OMFIT_REPO_URL:-git@github.com:wesleyliu728/OMFIT-source.git}"
OMFIT_DIR="${OMFIT_DIR:-OMFIT-source}"
ENV_FILE="${ENV_FILE:-environment.yml}"

# Always yes for conda
export CONDA_ALWAYS_YES=true
export PYTHONWARNINGS=ignore

echo "📦 Using env file: $ENV_FILE"

# ----------------------------
# Ensure conda is available
# ----------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda not found in PATH. Install Miniforge/Conda and retry."
  exit 1
fi

# Activate conda shell functions
source "$(conda info --base)/etc/profile.d/conda.sh"

# ----------------------------
# Read env name from environment.yml
# ----------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ $ENV_FILE not found."
  exit 1
fi

ENV_NAME="$(awk '/^name:/ {print $2}' "$ENV_FILE")"
if [[ -z "${ENV_NAME:-}" ]]; then
  echo "❌ Could not parse 'name:' from $ENV_FILE"
  exit 1
fi
echo "🧪 Target conda env: $ENV_NAME"

# ----------------------------
# Create or update env (non-interactive)
# Prefer: update if exists; else create
# ----------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "🔁 Updating existing env '$ENV_NAME' from $ENV_FILE..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune -y
else
  echo "🆕 Creating env '$ENV_NAME' from $ENV_FILE..."
  conda env create -f "$ENV_FILE" -y
fi

# ----------------------------
# Activate env
# ----------------------------
echo "🔌 Activating env: $ENV_NAME"
conda activate "$ENV_NAME"

# ----------------------------
# Optional: enforce Python/NumPy pins for legacy builds
# (Uncomment if you need NumPy<2 & Python 3.11)
# ----------------------------
# conda install -c conda-forge "python>=3.11,<3.12" "numpy<2" -y

# ----------------------------
# Toolchain & libs (one transaction)
# ----------------------------
echo "🧰 Installing compilers & libs..."
conda install -c conda-forge -y \
  compilers fortran-compiler make cmake \
  hdf5 netcdf4 openblas lapack fftw \
  cython "setuptools<70" pip

# ----------------------------
# Clone or update OMFIT repo/branch
# ----------------------------
if [[ -d "$OMFIT_DIR/.git" ]]; then
  echo "📁 Repo exists. Fetching & switching to '$OMFIT_BRANCH'..."
  git -C "$OMFIT_DIR" fetch --all --prune
  git -C "$OMFIT_DIR" checkout "$OMFIT_BRANCH"
  git -C "$OMFIT_DIR" pull --rebase
else
  echo "🔽 Cloning $OMFIT_REPO_URL (branch: $OMFIT_BRANCH) -> $OMFIT_DIR"
  git clone --branch "$OMFIT_BRANCH" "$OMFIT_REPO_URL" "$OMFIT_DIR"
fi

# ----------------------------
# Poetry setup (use conda env; no venv)
# ----------------------------
if ! command -v poetry >/dev/null 2>&1; then
  echo "📦 Installing Poetry into conda env..."
  conda install -c conda-forge poetry -y
fi

echo "⚙️  Configuring Poetry to use current conda env..."
poetry config virtualenvs.create false

# If your Python deps are in the repo root's pyproject.toml, run here.
# Otherwise cd into the correct project directory before installing.
echo "📦 Installing Python deps with Poetry..."
poetry install --no-interaction --no-ansi

# (Optional) install local OMFIT classes directly from your repo checkout:
# pip install -e "$OMFIT_DIR/omfit"

echo "✅ Setup complete. Active env: $ENV_NAME"
