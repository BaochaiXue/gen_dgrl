#!/bin/bash
set -euo pipefail

# Default locations for dataset and checkpoints
DATASET_DIR="${DATASET_DIR:-$(pwd)/data/expert_data_1M}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(pwd)/checkpoints/iql}"

# Control parallelism when downloading and extracting
N_DOWNLOAD_WORKERS="${N_DOWNLOAD_WORKERS:-4}"
# Use a single extraction worker by default to reduce memory usage
N_EXTRACT_WORKERS="${N_EXTRACT_WORKERS:-1}"

# Ensure dataset and checkpoint directories exist
mkdir -p "$DATASET_DIR" "$CHECKPOINT_DIR"

# Step 1: Download the Procgen expert dataset (1M transitions)
#python download.py --download_folder "$DATASET_DIR" --category_name 1M_E \
#  --clear_archives_after_unpacking \
#  --n_download_workers "$N_DOWNLOAD_WORKERS" \
#  --n_extract_workers "$N_EXTRACT_WORKERS"
# Step 2: Create a temporary config based on final/iql.json with paths filled in
CONFIG_DIR="configs/offline/final"
ORIG_CONFIG="$CONFIG_DIR/iql.json"
# Create a config file in the same directory that make_cmd expects
PIPELINE_CONFIG="$CONFIG_DIR/iql_pipeline.json"
# Ensure no stale config is present
rm -f "$PIPELINE_CONFIG"
python - "$ORIG_CONFIG" "$PIPELINE_CONFIG" "$DATASET_DIR" "$CHECKPOINT_DIR" <<'PY'
import json, sys
orig, dest, data_dir, ckpt_dir = sys.argv[1:]
with open(orig) as f:
    cfg = json.load(f)
cfg['grid']['dataset'] = [data_dir]
cfg['grid']['save_path'] = [ckpt_dir]
with open(dest, 'w') as f:
    json.dump(cfg, f, indent=4)
PY

# Step 3: Generate training commands for all environments
python -m train_scripts.make_cmd \
  --base_config offline \
  --dir final \
  --checkpoint \
  --grid_config iql_pipeline \
  --num_trials 1 \
  --module_name offline.train_offline_agent > pipeline_commands.txt

# Step 4: Run training sequentially
while read -r cmd; do
  [ -n "$cmd" ] || continue
  echo "Running: $cmd"
  eval "$cmd"
done < pipeline_commands.txt

# Step 5: Evaluate each trained model
while read -r cmd; do
  [ -n "$cmd" ] || continue
  eval_cmd=${cmd/offline.train_offline_agent/offline.evaluate_offline_agent}
  echo "Evaluating: $eval_cmd"
  eval "$eval_cmd"
done < pipeline_commands.txt

# Cleanup temporary config
rm -f "$PIPELINE_CONFIG"
