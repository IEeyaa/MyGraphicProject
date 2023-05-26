# -*- coding:utf-8 -*-
# @Author: IEeya
"""
    该部分用于草图从png、svg等格式的文件转变成合适的格式
"""
import math
from xml.dom import minidom


class Stroke:
    def __init__(self, stroke_points):
        """
        初始化Stroke对象。

        参数：
        - stroke_points：表示折线的点坐标列表

        Stroke对象包含以下属性：
        - pointNumber：折线的点数量
        - coordinates：折线的点坐标列表
        - length：折线的长度

        """
        self.pointNumber = len(stroke_points)
        self.coordinates = stroke_points
        self.length = self.calculate_length()

    def calculate_length(self):
        """
        计算折线的长度。

        返回值：
        - length：折线的长度

        """
        length = 0.0
        for i in range(len(self.coordinates) - 1):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[i + 1]
            segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            length += segment_length
        return length


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

    # 获取所有的polyline元素
    polyline_elements = graphInfo.getElementsByTagName("polyline")

    # 创建存储Stroke对象的列表
    strokes = []

    # 遍历每个polyline元素
    for polyline in polyline_elements:
        # 获取polyline的点坐标信息
        points_str = polyline.getAttribute("points")
        points = []
        coordinates = points_str.split(" ")
        for i in range(0, len(coordinates), 2):
            x = float(coordinates[i])
            y = float(coordinates[i + 1])
            points.append((x, y))
        # 创建Stroke对象并添加到列表中
        stroke = Stroke(points)
        strokes.append(stroke)

    # 打印每个Stroke对象的信息
    for stroke in strokes:
        print("Stroke: ", stroke.pointNumber, stroke.length)


if __name__ == '__main__':
    get_sketch_from_image("./data/sketch.svg")
