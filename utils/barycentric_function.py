import numpy as np

def barycentric_coords(p, a, b, c):
    """
    Calculate the barycentric coordinates of point p with respect to triangle (a, b, c).
    p, a, b, c are 2D points.
    Returns barycentric coordinates (u, v, w) where u + v + w = 1.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return u, v, w
