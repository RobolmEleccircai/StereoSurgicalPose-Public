"""Camera-relative surgical-pose decoding and video overlays; no Isaac Sim."""
__all__ = ['StereoFrame', 'PosePrediction', 'StereoOverlay', 'CppStereoOverlay', 'decode', 'project']


def __getattr__(name):
    if name == 'CppStereoOverlay':
        from .cpp_backend import CppStereoOverlay
        return CppStereoOverlay
    if name in ('StereoFrame', 'PosePrediction', 'StereoOverlay'):
        from . import pipeline
        return getattr(pipeline, name)
    if name == 'decode':
        from .decoding import decode
        return decode
    if name == 'project':
        from .projection import project
        return project
    raise AttributeError(name)
