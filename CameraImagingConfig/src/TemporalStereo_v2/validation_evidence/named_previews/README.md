# Named annotations of existing captures

These five examples were rebuilt from the saved RGB PNGs and per-eye ground truth with `tools/reannotate.py`. No simulator or renderer was started. All 32 existing pairs were reannotated in the review output; five representative pairs are backed up here. `annotation_report.json` records the 32 output hashes and confirms that all 224 source data files remained unchanged.

The yellow arrow follows positive shaft direction d, through C, from C−9d to C+6d in millimeters. Its endpoint is a drawing limit, not an anatomical landmark. C (shaft reference) and T (virtual tip midpoint) are diamonds and carry no surface-visibility assertion. J1/J2 name the two actual jaw endpoints. Filled circles mean visible; hollow circles and dashed leaders mean self-occluded; hollow crossed circles and dashed leaders mean tissue-occluded. Closed-jaw endpoint projections overlap because the original coordinates are retained.

- [Closed jaws](seq_000_validation_000.jpg)
- [Maximum opening](seq_000_validation_001.jpg)
- [Contact hold](seq_003_validation_030.jpg)
- [Surface sliding](seq_012_validation_026.jpg)
- [Tissue occlusion](seq_006_validation_014.jpg)
