# Named annotations of existing captures

These five examples were rebuilt from the saved RGB PNGs and per-eye ground truth with `tools/reannotate.py`. No simulator or renderer was started. All 32 existing pairs were reannotated in the review output; five representative pairs are backed up here. `annotation_report.json` records the 32 output hashes and confirms that all 224 source data files remained unchanged.

The yellow arrow starts exactly at C and follows positive shaft direction d to C+15d in millimeters. The 15 mm length and endpoint are drawing choices, not anatomical landmarks. T is marked as the midpoint of the two physical tips by a prominent magenta crosshair, with no direction arrow. C is a cyan diamond; neither C nor T carries a surface-visibility assertion. J1/J2 name the two actual jaw endpoints. Filled circles mean visible; hollow circles and dashed leaders mean self-occluded; hollow crossed circles and dashed leaders mean tissue-occluded. Closed-jaw endpoint projections overlap because the original coordinates are retained.

The latest previews show T's projected pixel coordinates beside its crosshair and list each eye's C/T camera coordinates, pixel coordinates and shaft direction below the image. All coordinates are recomputed from saved 3D truth and K/D; maximum discrepancy from saved pixels is 1.1368683772161603e-13 pixels. This is numerical consistency, not model accuracy. Previews explicitly identify the results as simulation-ground-truth projections, not model inference. Pixel coordinates exclude the additional title/footer margins.

- [Closed jaws](seq_000_validation_000.jpg)
- [Maximum opening](seq_000_validation_001.jpg)
- [Contact hold](seq_003_validation_030.jpg)
- [Surface sliding](seq_012_validation_026.jpg)
- [Tissue occlusion](seq_006_validation_014.jpg)
