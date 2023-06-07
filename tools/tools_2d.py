# -*- coding:utf-8 -*-
# @Author: IEeya
import numpy as np
from shapely import Point


# 返回集群后的交点坐标
def get_intersection_info(line_a, line_b):
    intersection = line_a.intersection(line_b)
    intersection_points = []
    if intersection.is_empty:
        return []  # 如果没有交点，返回空列表

    if intersection.geom_type == 'Point':
        intersection_points = [intersection]  # 如果只有一个交点，返回包含该交点的列表

    # 问题，如果两条螺旋线怎么办，但是可以保证的是如果不是曲线的话他们肯定不会相交？
    if intersection.geom_type == 'MultiPoint':
        intersection_points = [point.coords[0] for point in intersection.geoms]

    params_a = [line_a.project(Point(p), normalized=True) for p in intersection_points]
    params_b = [line_b.project(Point(p), normalized=True) for p in intersection_points]
    mid_param_a = (min(params_a) + max(params_a)) / 2
    mid_param_b = (min(params_b) + max(params_b)) / 2
    params_info = [eval_line(line_a, mid_param_a), [mid_param_a, mid_param_b]]
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
