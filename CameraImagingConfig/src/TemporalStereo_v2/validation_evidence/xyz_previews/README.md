# Per-camera XYZ annotation verification

The existing 32 stereo pairs were reannotated offline. All 224 source files remain unchanged; no simulation or RGB rendering was started.

Each eye displays C, J1 and J2 positions in its own optical frame (millimeters), d as a dimensionless unit direction, and separate J1-C / J2-C arrow displacements. Labels specify X right, Y down, Z forward. Position values use two displayed decimal places and directions use four; underlying data and projections retain full precision. Existing arrow endpoints and visibility labels are unchanged.

The report records all 32 results and code/font hashes. All displayed inputs and displacement vectors were checked against the source JSON, all output sizes are 2560 by 1140, and the maximum projection discrepancy is below 1e-6 pixels. The five examples cover closed jaws, maximum opening, contact hold, contact sliding and tissue occlusion. These are simulation truth, not model predictions.
