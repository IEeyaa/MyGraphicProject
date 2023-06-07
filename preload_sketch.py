# -*- coding:utf-8 -*-
# @Author: IEeya
"""
    该部分用于草图从png、svg等格式的文件转变成合适的格式
"""
import matplotlib.pyplot as plt
import numpy as np

from camera import Camera
from gather_block import gather_block_from_symmetric_lines
from get_candidate import get_candidates_from_same_dir, get_candidates_from_dif_dir, get_all_candidates_from_sketch
from pylowstroke.sketch_camera import assignLineDirection
from select_candidate import select_best_candidates


def init_camera(sketch):
    return get_camera(sketch)


def preload_sketch(sketch):
    # 过滤短线
    sketch.filter_strokes()
    # 标定身份
    sketch.judge_lines(5)

    # 加载相机
    cam = init_camera(sketch)
    # 测试，消除曲线
    curve = []
    for item in sketch.strokes:
        if item.is_curved:
            curve.append(item.id)
    sketch.delete_stroke_by_index(curve)
    # 计算相交关系
    sketch.get_intersect_map()
    # 组成直线集群
    sketch.get_line_cluster(10.0, 20.0, 5)
    # 集群优化
    sketch.generate_line_from_cluster()
    # 二次计算相交关系
    sketch.get_intersect_map()
    # 形成相交群
    sketch.get_intersect_info()

    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 形成block
    blocks= gather_block_from_symmetric_lines(candidate)
    # 进入最终筛查
    print(blocks)
    final_answer = select_best_candidates(cam, sketch, candidate, blocks)

    # 可视化
    visualize_lines(sketch)
    return cam, sketch


def get_camera(sketch):
    cam_param, line_groups = sketch.estimate_camera()
    cam = Camera(proj_mat=cam_param["P"],
                 focal_dist=cam_param["f"],
                 fov=cam_param["fov"],
                 t=cam_param["t"].reshape(-1),
                 view_dir=cam_param["view_dir"],
                 principal_point=cam_param["principal_point"].reshape(-1),
                 rot_mat=cam_param["R"],
                 K=cam_param["K"],
                 cam_pos=np.array(cam_param["C"]).reshape(-1),
                 vanishing_points_coords=cam_param["vp_coord"])
    cam.compute_inverse_matrices()
    assignLineDirection(sketch, line_groups)
    return cam


def visualize_lines(sketch):
    fig, ax = plt.subplots()
    strokes = sketch.strokes

    # 定义颜色映射
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

    for stroke in strokes:
        # 将LineString的坐标提取为NumPy数组形式
        line_string = stroke.lineString
        x, y = zip(*line_string.coords)

        # 根据axis_label选择颜色
        color = color_map.get(stroke.axis_label, 'black')

        # 绘制LineString
        ax.plot(x, y, marker='', linestyle='-', linewidth=1, color=color)

        # 添加文本标签显示线的编号
        ax.text((x[0]+x[-1])/2, (y[0]+y[-1])/2, str(stroke.id), fontsize=8, verticalalignment='bottom')

    ax.set_aspect('equal', adjustable='box')
    plt.show()
