#!/usr/bin/env bash
set -euo pipefail
pose_root=/home/leorobot/PoseProject/CameraImagingConfig/ShaftTipStereo_2026-11-07
refinement=/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06
pose_python=/home/leorobot/anaconda3/envs/grasp5_pixel_gpu/bin/python
capture_pid=${1:?Provide the live capture PID}
while ! grep -q '"completed": true' "$pose_root/render/native/manifest.json" 2>/dev/null; do
    if ! kill -0 "$capture_pid" 2>/dev/null; then
        echo 'CAPTURE_EXITED_BEFORE_COMPLETION' >&2
        exit 1
    fi
    sleep 10
done
"$pose_python" "$refinement/apply_initial_sensor_effects.py" --native "$pose_root/render/native" --out "$pose_root/render"
"$pose_python" "$refinement/DarkBorder_v9/generate.py" --source "$pose_root/render" --out "$pose_root/final"
"$pose_python" "$pose_root/package_dataset.py"
