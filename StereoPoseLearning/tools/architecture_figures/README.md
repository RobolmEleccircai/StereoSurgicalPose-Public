# Rebuild the architecture figures

These scripts draw reviewed model structure with shared PDF/SVG coordinates. They do not load weights, run training, or render the simulator. Committed figures are the approved publication artifacts; regeneration is optional.

From `StereoPoseLearning/` in a document environment:

```bash
python -m pip install -r tools/architecture_figures/requirements.txt
# Ubuntu: install fonts-liberation2 and poppler-utils using the system package manager.
python tools/architecture_figures/draw_framework.py --output /tmp/pose-figures
python tools/architecture_figures/draw_details.py --output /tmp/pose-figures
pdftoppm -r 240 -singlefile -png /tmp/pose-figures/framework.pdf /tmp/pose-figures/framework
pdftoppm -r 150 -png /tmp/pose-figures/detailed_structure.pdf /tmp/pose-figures/detail
```

If the fonts are elsewhere, pass `--font-dir /path/to/fonts` or set `LIBERATION_FONT_DIR`. The directory must contain `LiberationSans-Regular.ttf` and `LiberationSans-Bold.ttf`. Both faces are embedded in the main SVG/PDF. Font substitution is not silently accepted. The detailed figures preserve their original Helvetica-based typography and layout.

`--assets` overrides the framework's example asset directory; `--audit` overrides the detailed figure's shape-audit JSON. Defaults resolve relative to the repository, with no user-specific paths. Current source examples contain one already-existing image pair and predicted maps/crops, not new generated data. PNGs are GitHub previews, SVGs retain editable vectors/text, and PDFs are for publication. PDF timestamps can differ on regeneration; compare content, dimensions and rendered output rather than assuming byte-identical PDFs.

Reviewed environment: ReportLab 4.4.9, Pillow 13.0.0, Liberation Sans regular/bold, Poppler. These document-only dependencies do not alter the model's training/inference environment.
