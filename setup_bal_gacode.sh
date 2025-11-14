#!/bin/bash
set -e  # exit on error
set -x  # print commands for debugging

# -----------------------------
# 0️⃣ Define absolute paths
# -----------------------------
PROJECT_ROOT="$HOME/ai-fusion-cgyro-nn"          # Training pipeline repo
SCHEDULER_ROOT="$HOME/ai-fusion-bal-scheduler"  # Scheduler repo
GACODE_ROOT="$HOME/gacode"                       # Simulation binaries
GACODE_DOCKER_ROOT="$HOME/gacode-docker"        # Platform files repo

# -----------------------------
# 1️⃣ Clone or update gacode-docker (platform files)
# -----------------------------
if [ ! -d "$GACODE_DOCKER_ROOT" ]; then
    git clone --depth=1 --branch main https://github.com/zclawr/gacode-docker.git "$GACODE_DOCKER_ROOT"
else
    echo "gacode-docker already exists, updating..."
    cd "$GACODE_DOCKER_ROOT"
    git pull
    git submodule update --init --recursive
fi

# -----------------------------
# 2️⃣ Clone or update ai-fusion-bal-scheduler
# -----------------------------
if [ ! -d "$SCHEDULER_ROOT" ]; then
    git clone --depth=1 --branch docker-fix https://github.com/zclawr/ai-fusion-bal-scheduler.git "$SCHEDULER_ROOT"
else
    echo "ai-fusion-bal-scheduler already exists, updating..."
    cd "$SCHEDULER_ROOT"
    git checkout docker-fix
    git pull
    git submodule update --init --recursive
fi

# -----------------------------
# 3️⃣ Run output_parsing setup
# -----------------------------
OUTPUT_PARSING_DIR="$SCHEDULER_ROOT/src/output_parsing"
cd "$OUTPUT_PARSING_DIR"

# Fix missing README.md to avoid Poetry warnings
if [ ! -f README.md ]; then
    touch README.md
fi

bash setup.sh
cd "$HOME"

# -----------------------------
# 4️⃣ Clone or update GACODE
# -----------------------------
if [ ! -d "$GACODE_ROOT" ]; then
    git clone https://github.com/gafusion/gacode.git "$GACODE_ROOT"
else
    echo "gacode already exists, updating..."
    cd "$GACODE_ROOT"
    git pull
    git submodule update --init --recursive
fi

# -----------------------------
# 5️⃣ Copy platform files and run script
# -----------------------------
cp "$GACODE_DOCKER_ROOT/src/platform/exec.LINUX_DOCKER" "$GACODE_ROOT/platform/exec/exec.LINUX_DOCKER"
cp "$GACODE_DOCKER_ROOT/src/platform/make.inc.LINUX_DOCKER" "$GACODE_ROOT/platform/build/make.inc.LINUX_DOCKER"
cp "$GACODE_DOCKER_ROOT/src/run_simulation.sh" "$HOME/run_simulation.sh"
chmod +x "$HOME/run_simulation.sh"

# -----------------------------
# 6️⃣ Set environment variables for this session
# -----------------------------
export GACODE_PLATFORM=LINUX_DOCKER
export GACODE_ROOT="$GACODE_ROOT"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# -----------------------------
# 7️⃣ Build TGLF and CGYRO
# -----------------------------
source "$GACODE_ROOT/shared/bin/gacode_setup"

cd "$GACODE_ROOT/cgyro"
make
cd ../tglf
make
cd "$HOME"

echo "✅ Setup complete. GACODE binaries are ready."
