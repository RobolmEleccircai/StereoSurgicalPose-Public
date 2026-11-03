# Physical jaw-hinge annotation verification

All 32 existing stereo pairs were annotated offline. Shaft XYZ remains anchored at C. Each jaw uses its native body rotation, with origin at its authored USD body1/localPos1 joint anchor; blue/orange tip arrows start at H1/H2 and end at the unchanged physical J1/J2. Parent-side joint anchors independently confirm the same centers.

The two actual jaw centers coincide to USD floating-point precision (maximum separation 0.00000637 mm). Their distance from C is approximately 9.1 mm. No anchor or arrow endpoint is shifted for display. Coincident axis names may be combined; the underlying records remain separate.

Verification covers 192 local triads, 576 axis arrows and 128 hinge-to-tip arrows. It checks independent projections, native jaw orientations, parent/child joint closure, relative jaw rotation against saved opening angles, and rejection of a deliberately incorrect hinge. All 224 source image/mask/label/camera files and saved pose geometry remain unchanged. No SimulationApp or renderer was started.

`jaw_hinges.json` contains the source USD hashes and authored joint-frame transforms. `verification.json` records numerical errors and the annotation-code hash. The three examples show closed, fully open and contact-sliding states. These annotations use simulation ground truth, not model predictions.
