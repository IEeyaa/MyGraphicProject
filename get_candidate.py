# -*- coding:utf-8 -*-
# @Author: IEeya
# 该函数主要用于从Sketch中获取全部的候选组
from tools.tools_2d import get_projection_info


def get_all_candidates_from_sketch(sketch, cam):
    # 主视角
    main_dir = [0, 1, 2]
    all_candidates = []
    # 遍历所有视角
    for vp_dir in main_dir:
        # 遍历所有线
        vanishing_point_dir = vp_dir
        for stroke in sketch.strokes:
            stroke_dir = stroke.axis_label
            # 相同朝向
            if stroke_dir == vanishing_point_dir:
                # 调用相关函数
                all_candidates.append(get_candidates_from_same_dir(cam, stroke, sketch))
            # 不同朝向
            else:
                # 调用相关函数
                all_candidates.append(
                    get_candidates_from_dif_dir(cam, vanishing_point_dir, stroke_dir, stroke, sketch.intersect_map))


# 获取所有的合理的交叉点组合以及其中点投影坐标
def get_candidates_from_same_dir(cam, stroke, sketch):
    # 获取
    main_axis = stroke.axis_label
    intersect_infos = sketch.intersect_infor[stroke.id]
    intersect_params_info = []
    # 得到所有的intersect_middle_params情况
    for intersect_info in intersect_infos:
        intersect_stroke_id = intersect_info.stroke_id[1]
        intersect_params_info.append([intersect_stroke_id, intersect_info.inter_params[2]])
    # 分组
    param_middle = 0.5
    deviation = 0.001
    # 间隔阈值，大于3/10的线段长度
    threshold = 0.3

    left_side = []
    right_side = []
    for param in intersect_params_info:
        if param[1] < param_middle - deviation:
            left_side.append(param)
        elif param[1] > param_middle + deviation:
            right_side.append(param)

    candidates = [[left_item, right_item]
                  for left_item in left_side
                  for right_item in right_side
                  if sketch.strokes[left_item[0]].axis_label == sketch.strokes[right_item[0]].axis_label and
                  abs(left_item[1] - right_item[1]) > threshold
                  ]
    # 判断，形成candidate
    return candidates


def get_candidates_from_dif_dir(cam, vp_dir, s_dir, stroke, intersect):
    print(1)
