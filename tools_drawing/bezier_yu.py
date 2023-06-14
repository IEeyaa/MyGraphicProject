import numpy as np
import math

def binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(
        math.factorial(i) * math.factorial(n - i))


def bernstein(t, i, n):
    """Bernstein polynom"""
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(ctrl_points, t):
    """Calculate coordinate of a point in the bezier curve"""
    n = len(ctrl_points) - 1
    x = y = z = 0
    for i, pos in enumerate(ctrl_points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
        #z += pos[2] * bern
    return x, y#, z

# CALL THIS (and tweak parameters maybe)
def get_closest_t(ctrl_pts, position):
    return r_get_closest_t(ctrl_pts, position, 50, 0, 1, 30)

# Recursively find the parameter of the closest point on polybezier curve from a given position
def r_get_closest_t(ctrl_pts, position, n_slices, start, end, iterations):
    if (iterations <= 0):
        return (start + end) / 2
    step = (end - start) / n_slices
    if (step < 10e-10):
        return (start + end) / 2
    tMin = 0
    t = start
    dMin = float('inf')

    while (t <= end):
        d = np.linalg.norm(np.hstack(bezier(ctrl_pts, t)) - position)
        if (d < dMin):
            tMin = t
            dMin = d
        t += step
    new_start = max(tMin - step, 0)
    new_end = min(tMin + step, 1)
    new_iterations = iterations - 1
    return r_get_closest_t(ctrl_pts, position, n_slices, new_start, new_end, new_iterations)
