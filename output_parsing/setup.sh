#!/bin/bash

# Exit on error
set -e

# Variables
OMFIT_BRANCH="xarray-fix"
OMFIT_REPO_URL="git@github.com:wesleyliu728/OMFIT-source.git" # Replace with your fork
OMFIT_DIR="OMFIT-source"

# Step 1: Create or update conda environment
echo "📦 Creating or updating conda environment from environment.yml..."
conda env create -f environment.yml || conda env update -f environment.yml

# Extract env name
ENV_NAME=$(awk '/name:/ {print $2}' environment.yml)

# Step 2: Activate the conda environment
echo "🔁 Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Step 3: Clone the OMFIT-source repository from the specific branch
if [ ! -d "$OMFIT_DIR" ]; then
    echo "🔽 Cloning OMFIT-source from branch '$OMFIT_BRANCH'..."
    git clone --branch "$OMFIT_BRANCH" "$OMFIT_REPO_URL" "$OMFIT_DIR"
else
    echo "📁 Directory '$OMFIT_DIR' already exists. Skipping clone."
fi

conda install -c conda-forge hdf5
conda install -c conda-forge netcdf4
# Compilers (C, C++, Fortran) + build tools
conda install -c conda-forge compilers make cmake

# Linear algebra / math libraries
conda install -c conda-forge openblas lapack fftw

# Step 4: Poetry install
echo "📦 Installing dependencies using Poetry..."
poetry install

echo "✅ Setup complete. You're now in Conda environment '$ENV_NAME'."