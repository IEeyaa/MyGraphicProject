# -*- coding:utf-8 -*-
# @Author: IEeya

# for anchor
# 判断每一个stroke的锚定情况
def get_anchor_from_intersection(sketch):
    threshold = 0.1
    intersection = sketch.intersect_infor
    strokes = sketch.strokes
    stroke_number = len(strokes)
    intersection_dict = sketch.intersect_dict
    final_info = [[] for item in range(0, stroke_number)]
    for stroke in strokes:
        temp_info = []
        stroke_intersection_info = [intersection[index] for index in intersection_dict[stroke.id]]
        # 检查所有的intersection情况
        for intersect in stroke_intersection_info:
            # 分别代表0 1 2 3
            anchor_check = [0, 0, 0, 0]
            neighbors = [intersection[index] for index in intersect.adjacent_inter_ids]
            for neighbors_info in neighbors:
                anchor_check[strokes[neighbors_info.stroke_id[0]].axis_label] = 1
                anchor_check[strokes[neighbors_info.stroke_id[1]].axis_label] = 1
            # 良好锚定 high value 点
            if sum(anchor_check) > 2:
                temp_info.append(intersect.id)
        # 以下为个人改进，个人认为，两个良好锚定点应该位于线的两侧
        left_side = 0
        right_side = 0
        well_anchor_intersection = [intersection[index] for index in temp_info]
        for inter in well_anchor_intersection:
            if inter.stroke_id[0] == stroke.id:
                inter_param = inter.inter_params[0]
            elif inter.stroke_id[1] == stroke.id:
                inter_param = inter.inter_params[1]
            else:
                final_info[stroke.id] = []
                break
            if inter_param < 0.5 - threshold:
                left_side = 1
            elif inter_param > 0.5 + threshold:
                right_side = 1
        if left_side == 1:
            final_info[stroke.id].append([1])
        if right_side == 1:
            final_info[stroke.id].append([2])
    return final_info


# for coverage
# 对于每一个stroke, 根据其交点生成最近和最远
def get_coverage_params_for_strokes(sketch):
    coverage_params = [[] for i in range(len(sketch.strokes))]
    for inter in sketch.intersect_infor:
        stroke_ids = inter.stroke_id
        params = inter.inter_params
        coverage_params[stroke_ids[0]].append(params[0])
        coverage_params[stroke_ids[1]].append(params[1])
    for index, item in enumerate(coverage_params):
        # 没有最大最小
        if len(item) < 2:
            coverage_params[index] = [0, 1]
        else:
            min_param = min(item)
            max_param = max(item)
            coverage_params[index] = [min_param, max_param]
    return coverage_params


# 根据stroke对candidate进行分类
def get_candidate_by_stroke(candidate, sketch):
    strokes = sketch.strokes
    stroke_number = len(strokes)
    candidate_of_stroke = [[] for i in range(stroke_number)]
    for index, item in enumerate(candidate):
        candidate_of_stroke[item[0]].append(index)
        if item[0] != item[1]:
            candidate_of_stroke[item[1]].append(index)
    return candidate_of_stroke


