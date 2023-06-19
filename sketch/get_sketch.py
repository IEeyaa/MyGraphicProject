# -*- coding:utf-8 -*-
# @Author: IEeya
"""
    该部分用于草图从png、svg等格式的文件转变成合适的格式
"""
from xml.dom import minidom

import numpy as np
from shapely.geometry import LineString
from svgpathtools import parse_path

from sketch.sketch_info import Stroke, Sketch


def get_sketch_from_image(filepath):
    """
    从SVG图像文件中提取草图信息。

    参数：
    - filepath：SVG图像文件的路径

    返回值：
    - strokes：包含草图信息的Stroke对象列表

    Stroke对象包含以下属性：
    - points：表示草图折线的点坐标列表
    - pointNumber：折线的点数量
    - length：折线的长度

    """

    # 解析SVG文件
    graphInfo = minidom.parse(filepath)

    # 获取viewBox属性值
    viewBox = graphInfo.documentElement.getAttribute('viewBox')

    # 提取宽度和高度
    values = viewBox.split(' ')
    if len(values) > 4:
        width = values[2]
        height = values[3]
    else:
        width = 972
        height = 972

    # 获取所有的polyline元素
    polyline_elements = graphInfo.getElementsByTagName("polyline")

    path_elements = graphInfo.getElementsByTagName("path")

    line_elements = graphInfo.getElementsByTagName("line")

    # 创建存储Stroke对象的列表
    strokes = []

    # 遍历每个polyline元素
    for index, polyline in enumerate(polyline_elements):
        # 获取polyline的点坐标信息
        points_str = polyline.getAttribute("points")
        points = []
        coordinates = points_str.split(" ")
        for i in range(0, len(coordinates), 2):
            x = float(coordinates[i])
            y = float(coordinates[i + 1])
            points.append((x, y))
        # 创建Stroke对象并添加到列表中
        stroke = Stroke(LineString(points))
        strokes.append(stroke)

    for index_path, path_data in enumerate(path_elements):
        # 获取path的点坐标信息
        path = parse_path(path_data.getAttribute("d"))
        if len(path) == 0:
            continue
        if path.start == path.end:
            continue
        coordinates = [path.point(t) for t in np.linspace(0, 1.0, 10)]
        # 解析path信息，提取点坐标
        points = [(c.real, c.imag) for c in coordinates]
        # 创建Stroke对象并添加到列表中
        stroke = Stroke(LineString(points))
        strokes.append(stroke)

    for index_line, line_data in enumerate(line_elements):
        # 获取line的起始点和结束点坐标
        x1 = float(line_data.getAttribute("x1"))
        y1 = float(line_data.getAttribute("y1"))
        x2 = float(line_data.getAttribute("x2"))
        y2 = float(line_data.getAttribute("y2"))

        # 创建包含起始点和结束点的列表
        points = [(x1, y1), (x2, y2)]

        # 创建Stroke对象并添加到列表中
        stroke = Stroke(LineString(points))
        strokes.append(stroke)

    # 将信息存储到Sketch中
    initial_sketch = Sketch(float(width), float(height), 1, strokes)
    return initial_sketch


if __name__ == '__main__':
    sketch = get_sketch_from_image("../data/sketch.svg")
