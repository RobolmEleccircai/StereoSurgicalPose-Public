# Model input and output

The current [17-value contract and fixed kinematics](../../../StereoPoseLearning/docs/ARCHITECTURE.md) define model input, output, coordinate frames and label usage. The [detailed diagrams](../../../StereoPoseLearning/docs/MODEL_DIAGRAMS.md) show tensor dimensions and the stereo/temporal branches.

Generation records per-eye camera-frame truth, including shaft reference C, shaft direction d, physical hinge centers H1/H2, jaw endpoints J1/J2, their midpoint T and complete part rotations. The network predicts shaft pose and shared articulation; fixed kinematics computes the remaining quantities. Relative camera extrinsics remain separate evaluation truth, not a required model input.
