# Relocated bundle audit — 2026-11-08

`relocated_preflight.json` was produced by `tools/server_preflight.py` on a new copy of the code/assets under `server_transfer_20261108`. The machine retained the `laptop` role. FCL was installed from the included CPython 3.11 / Linux x86_64 wheel into the transfer copy's isolated dependency directory; the original environment was not changed.

The check verified all 76 asset-manifest entries, 78 active dependency files, 17 anatomical meshes, 12 moving meshes and 15 closed contact surfaces. Live references resolve only inside the transferred assets or the pinned Isaac Sim installation. It exercised a native FCL collision query, loaded the bundled font and encoded/read back two synthetic color frames to check MP4 support. No simulated scene images were rendered and no full trajectory-generation or capture job was started.

The report is evidence of relocation and dependency readiness on the existing laptop, **not a target-server GPU/rendering certification**. Server driver/RTX compatibility and a server render check remain pending. The two synthetic codec-test frames are temporary test files, not dataset samples; the temporary directory is removed by the check.

Existing source assets and previous datasets were not changed. The package's top-level `MANIFEST.sha256`, `release.json`, and adjacent archive SHA-256 identify the delivered files separately from this preflight report.
