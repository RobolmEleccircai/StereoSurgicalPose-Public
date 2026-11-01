#!/usr/bin/env bash
set -euo pipefail
temporal_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
isaac_python=/home/leorobot/isaacsim-5.1.0/python.sh
export LD_LIBRARY_PATH="/home/leorobot/isaacsim-5.1.0/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$temporal_dir/.run.lock"
flock -n 9 || { echo 'An existing complete-dataset run is active.'; exit 1; }
"$isaac_python" "$temporal_dir/prepare.py" --config "$temporal_dir/config.json"
render_success=false
for attempt in 1 2 3; do
    if "$isaac_python" "$temporal_dir/capture.py" --config "$temporal_dir/config.json" >>"$temporal_dir/render.log" 2>&1; then
        render_success=true
        break
    fi
    echo "Render process exited; resuming incomplete frames (attempt $attempt of 3)."
done
if [ "$render_success" != true ]; then
    echo 'Render failed after three process attempts. See render.log; completed images are preserved.'
    exit 1
fi
"$isaac_python" "$temporal_dir/finalize.py" --config "$temporal_dir/config.json"
