#!/bin/bash
# Run the polygon-crossing detection app.
# All arguments are forwarded to detection.py.
# Example:
#   ./run_polygon_crossing.sh --config detection_config_example.yaml --input rpi

# Resolve the repo root 
REPO_ROOT="/home/parkomat/hailo-apps"

# Change to the repo root so setup_env.sh works correctly
cd "$REPO_ROOT" || { echo "Failed to cd to $REPO_ROOT"; exit 1; }

# Source the environment (activates venv + sets PYTHONPATH)
source setup_env.sh || { echo "Failed to set up environment"; exit 1; }

# Run the detection app, forwarding all arguments
python3 hailo_apps/python/pipeline_apps/detection/detection.py --config hailo_apps/python/pipeline_apps/detection/detection_config_example.yaml -i rpi
