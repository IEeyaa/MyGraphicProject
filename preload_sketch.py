# -*- coding:utf-8 -*-
# @Author: IEeya
"""
    该部分用于草图从png、svg等格式的文件转变成合适的格式
"""
import matplotlib.pyplot as plt


def preload_sketch(sketch):
    # 过滤短线
    sketch.filter_strokes()
    # 标定身份
    sketch.judge_lines(5)
    # 测试，消除曲线
    curve = []
    for item in sketch.strokes:
        if item.type == "curve":
            curve.append(item.id)
    sketch.delete_stroke_by_index(curve)
    # 计算相交关系
    sketch.get_intersect_map()
    # 组成直线集群
    sketch.get_line_cluster(10.0, 20.0, 5)
    # 集群优化

    # 可视化
    visualize_lines(sketch)


def visualize_lines(sketch):
    fig, ax = plt.subplots()
    strokes = sketch.strokes
    for stroke in strokes:
        # 将LineString的坐标提取为NumPy数组形式
        line_string = stroke.lineString
        x, y = zip(*line_string.coords)

        # 绘制LineString
        ax.plot(x, y, marker='o', linestyle='-', linewidth=1)

        # 添加文本标签显示线的编号
        ax.text(x[0], y[0], str(stroke.id), fontsize=8, verticalalignment='bottom')

    ax.set_aspect('equal', adjustable='box')
    plt.show()

