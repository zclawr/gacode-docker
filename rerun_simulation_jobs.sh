#!/bin/bash

if [[ -z "$1" ]]; then
  echo "Usage: $0 runs/YYYYMMDD_HHMMSS/"
  exit 1
fi

RUN_DIR="$1"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "❌ Directory $RUN_DIR does not exist."
  exit 1
fi

YAML_FILES=("$RUN_DIR"/launch_*.yaml)

if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
  echo "❌ No launch_*.yaml files found in $RUN_DIR"
  exit 1
fi

for yaml in "${YAML_FILES[@]}"; do
  echo "🔁 Re-running job using $yaml"
  cp "$yaml" ./config/launch.yaml
  make job overwrite=True
done
