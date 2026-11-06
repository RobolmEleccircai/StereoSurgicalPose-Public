"""Build physical arrows and displaced XYZ panels from camera-relative poses."""
import numpy as np
from .projection import project

XYZ_MAX_PROJECTED_RATIO = .55


def build_geometry(poses, cameras, model):
    def curve(a, b, eye):
        pixels = project(np.linspace(a, b, 61), cameras[eye]['K'], cameras[eye]['D'])
        if not np.isfinite(pixels).all() or abs(pixels).max() > 1e6:
            raise ValueError('Projection is outside the supported drawing range')
        return pixels

    def length(points):
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    result = {}
    for eye, g in poses.items():
        C = g['C']; R = g['R_shaft']
        references = [length(curve(C, C+15*g['d'], eye)),
                      *(length(curve(C, g[f'J{i}'], eye)) for i in (1, 2))]
        result[eye] = {'anchor': 'C', 'frame': model['shaft_body'], 'axes': [],
                       'source': 'camera_relative_pose', 'jaw_frames': [],
                       'reference_px': max(12., min(references)),
                       'shared_hinge_within_1e_4_mm': bool(np.linalg.norm(g['H1']-g['H2']) < 1e-4)}
        for i in (1, 2):
            H, J = g[f'H{i}'], g[f'J{i}']
            line = curve(H, J, eye)
            result[eye]['jaw_frames'].append({'anchor': f'H{i}', 'jaw_index': i,
                'hinge_camera_mm': H.tolist(), 'hinge_pixel': project(H, cameras[eye]['K'], cameras[eye]['D']).tolist(),
                'tip_camera_mm': J.tolist(), 'hinge_to_tip_polyline_pixels': line.tolist(),
                'reference_px': max(12., length(line)), 'axes': []})

    # Preserve a common short physical display length in both eyes and both jaws.
    for group in ('shaft', 'jaws'):
        lengths = [min(15., *(np.linalg.norm(g[f'J{i}']-g['C']) for i in (1, 2)))
                   if group == 'shaft' else min(np.linalg.norm(g[f'J{i}']-g[f'H{i}']) for i in (1, 2))
                   for g in poses.values()]
        size = .5*min(lengths)
        if size <= 1e-9: raise ValueError('Degenerate arrow geometry')
        for _ in range(100):
            fits = True
            for eye, g in poses.items():
                specs = [(result[eye], g['C'], g['R_shaft'])] if group == 'shaft' else [
                    (result[eye]['jaw_frames'][i-1], g[f'H{i}'], g[f'R_jaw{i}']) for i in (1, 2)]
                for entry, origin, rotation in specs:
                    axes = []
                    for i, name in enumerate('XYZ'):
                        end = origin+size*rotation[:, i]; line = curve(origin, end, eye)
                        pixels = length(line)
                        fits &= pixels <= XYZ_MAX_PROJECTED_RATIO*entry['reference_px']
                        axes.append({'name': name, 'start_camera_mm': origin.tolist(),
                            'end_camera_mm': end.tolist(), 'direction_camera_unit': rotation[:, i].tolist(),
                            'projected_polyline_pixels': line.tolist(), 'projected_length_px': pixels})
                    entry.update(axes=axes, length_mm=float(size))
            if fits: break
            size *= .9
        else: raise ValueError('Unable to size XYZ panels')
    return result
