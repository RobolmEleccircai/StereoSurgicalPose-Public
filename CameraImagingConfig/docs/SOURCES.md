# Configuration sources

| Item | Source / provenance |
|---|---|
| Intrinsics, distortion and reference stereo geometry | [DRENDS dataset record](https://zenodo.org/records/17598453), ExVivo calibration |
| Calibration field definitions | [DRENDS calibration implementation](https://github.com/L0za007/DRENDS/blob/077ec79fd4c414508bea1de1ef1e14b94911c942/Calibration/stereo_calibration.py) |
| Surgical robot assets used by the source experiment | [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical) |
| Anatomy scene used by the source experiment | [SuFIA-BC project](https://orbit-surgical.github.io/sufia-bc/) |
| Rendering and camera interface | [Isaac Sim 5.1 camera documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html) |
| Vignette 3.5% and global brightness ×0.75 | Explicit user settings, visually accepted on 2026-11-07 |
| Ellipse, effective optics, blur and noise | Recorded simulation tuning retained in the accepted configuration |

The copied DRENDS ExVivo calibration JSON has SHA256 `bdb175e73c4976a9ca5c317ab6be7ae167e540889f7aa75444d57ef7ebc4652b`. Its parameters describe that published experimental setup, not universal dVRK factory settings. Original datasets and assets retain their upstream licensing and attribution requirements.

Only parameter records and a derived preview are included here. Original recordings, third-party USD asset bundles, render caches and trained weights are stored separately.
