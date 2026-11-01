# CAD pose projection overlays

These images show the original instrument CAD silhouette/boundary edges projected using the saved full joint states and per-eye camera truth. The export reads USD only and verifies C/T against saved labels. It never creates a SimulationApp or renders new RGB images. Nine actual visual meshes contain 448,894 source triangles. All 32 existing pairs were processed; five examples are included here.

Cyan is CAD projection over the visible instrument region. Interior lines have 20% opacity so they do not obscure the RGB; boundary-region lines have 95% opacity. The existing visible mask only sets opacity/display-region classification. It does not generate, relocate or fit projected edges. Magenta dashed lines lie outside that visible region. This wireframe display is not a new per-facet self-occlusion annotation.

The yellow arrow starts at C and follows positive shaft direction d. T is named by a leader at the projected tip midpoint, without a circle/crosshair. These full-CAD diagnostics additionally use saved joint states: they are not reconstructions of every articulated detail from C/d/T alone, and are not model inference. Target definitions remain unchanged.

`overlay_report.json` records zero new RGB renders and byte-identical preservation of all 224 original data files. Geometry and output hashes are retained. Full images and exported geometry remain in the run's review output.
