# -*- coding:utf-8 -*-
# @Author: IEeya
"""
    该部分用于草图从png、svg等格式的文件转变成合适的格式
"""
import matplotlib.pyplot as plt
import numpy as np

from sketch.camera import Camera
from pylowstroke.sketch_camera import assignLineDirection


def init_camera(sketch):
    return get_camera(sketch)


def preload_sketch(sketch):
    # 过滤短线
    sketch.filter_strokes()
    # 标定身份
    sketch.judge_lines(2)
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
    sketch.get_line_cluster(10.0, 5.0, 5)
    # 集群优化
    sketch.generate_line_from_cluster()
    # 二次计算相交关系
    sketch.get_intersect_map()
    # 形成相交群
    sketch.get_intersect_info()
    # 形成邻接表
    sketch.get_adjacent_intersections()
    # sketch.strokes[18].axis_label = 3

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


def visualize_lines(line_datas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制线段
    for line_data in line_datas:
        # line_data = line_data[0]
        for i in range(len(line_data) - 1):
            start_point = line_data[i]
            end_point = line_data[i + 1]
            print(start_point)
            print(end_point)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]])

    # 设置坐标轴范围
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_zlim([0, 4])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()
