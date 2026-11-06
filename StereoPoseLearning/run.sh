#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/reference"
if [ -n "${STEREO_POSE_PYTHON:-}" ]; then
    PYTHON_EXECUTABLE="$STEREO_POSE_PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_EXECUTABLE="$(command -v python)"
elif [ -f .python-path ]; then
    IFS= read -r PYTHON_EXECUTABLE < .python-path
else
    printf '%s\n' 'Activate the project Conda/venv environment or set STEREO_POSE_PYTHON.' >&2
    exit 1
fi
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    printf '%s\n' 'Activate the project environment or set STEREO_POSE_PYTHON to its Python interpreter.' >&2
    exit 1
fi
if [ "${1:-}" = prediction ]; then
    shift
    exec "$PYTHON_EXECUTABLE" -m stereo_pose.predict "$@"
elif [ "${1:-}" = video ]; then
    shift
    exec "$PYTHON_EXECUTABLE" -m pose_display.video --predictor stereo_pose.predict:build_predictor "$@"
else
    exec "$PYTHON_EXECUTABLE" -m stereo_pose "$@"
fi
