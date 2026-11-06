"""Projection into the saved raw distorted image geometry."""
import numpy as np

def project(points, K, D):
    """OpenCV five-coefficient distorted projection; valid only for positive Z."""
    p = np.asarray(points, float); K = np.asarray(K, float)
    if not np.isfinite(p).all() or np.any(p[..., 2] <= 0):
        raise ValueError('Cannot project a nonfinite point or a point behind the camera')
    x, y = (p[..., :2] / p[..., 2, None]).T
    k1, k2, p1, p2, k3 = D
    r2 = x*x + y*y; radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = x*radial + 2*p1*x*y + p2*(r2+2*x*x)
    yd = y*radial + p1*(r2+2*y*y) + 2*p2*x*y
    return np.stack((K[0, 0]*xd + K[0, 2], K[1, 1]*yd + K[1, 2]), axis=-1)


def project_point(p, K, D):
    # Scalar order retained for pixel-identical legacy annotations.
    x, y = p[0] / p[2], p[1] / p[2]
    k1, k2, p1, p2, k3 = D
    r = x*x + y*y
    radial = 1 + k1*r + k2*r*r + k3*r*r*r
    return (K[0][0]*(x*radial + 2*p1*x*y + p2*(r+2*x*x)) + K[0][2],
            K[1][1]*(y*radial + p1*(r+2*y*y) + 2*p2*x*y) + K[1][2])
