# -*- coding:utf-8 -*-
# @Author: IEeya
import numpy as np
from shapely import Point


def get_projection_info(line_a, line_b):
    params_a = [line_a.project(Point(p), normalized=True) for p in line_b.coords]
    params_b = [line_b.project(Point(p), normalized=True) for p in line_a.coords]
    min_param_a = min(params_a)
    max_param_a = max(params_a)
    min_param_b = min(params_b)
    max_param_b = max(params_b)
    mid_param_a = (min_param_a + max_param_a) / 2
    mid_param_b = (min_param_b + max_param_b) / 2
    params_info = [[eval_line(line_a, mid_param_a), eval_line(line_b, mid_param_b)],
                   [min_param_a, max_param_a, mid_param_a, min_param_b, max_param_b, mid_param_b]]
    return params_info


def eval_line(line, t):
    """
    Eval polyline at parameter t

    :param line: linestring
    :param t: parameter in [0 -> 1]
    :return: (x, y)
    """
    if t > 1.0 or t < 0.0:
        print("Error: cannot evaluate stroke at t=%s" % t)
        raise ValueError

    p = line.interpolate(t, normalized=True)
    return np.array([p.x, p.y])
