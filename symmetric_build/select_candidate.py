# -*- coding:utf-8 -*-
# @Author: IEeya

import numpy as np
from sklearn.cluster import MeanShift


def select_best_candidates(cam, sketch, candidates, blockes):
    """
        :param cam: 相机
        :param sketch: 草图类
        :param candidates:  待筛选的所有候选对称对
        :param blockes:     分类块
        :return: best_candidates:   最佳匹配

        这个函数主要用于从对应的candidates中选择合适的
        每个candidates都会产生一个三维重建的结果，这个结果保存在all_candidates中
        all_candidates内容: [s_id_0, s_id_1, 3d_res_0, 3d_res_1, focus_dir, from_id_0, from_id_1]
        s_id_0, 3d_res_0: 重建的第一条线id与线的3d坐标[p0, p1]
        s_id_1, 3d_res_1: 重建的第二条线id与线的3d坐标[p0, p1]
        focus_dir: 依据的重构平面方向，只可能属于[0, 1, 2]
        inter_id_0, inter_id_1: 重建依据(只可能在自身重建时使用)
    """
    # 对每一个block进行操作
    for block in blockes:
        candidates_info = gather_construction_from_dif_direction_per_block(candidates, block)
        plane_scale_factor = get_scale_factor_for_each_plane(cam, candidates_info, sketch)


def gather_construction_from_dif_direction_per_block(candidates, block):
    """
        :param candidates: 待筛选的所有候选对称对
        :param block:       当前处理的block
        :return: candidates_info:   block中candidates的重建情况
    """
    candidates_info = [{}, {}, {}]
    for item in candidates:
        if item[0] in candidates_info[item[4]].keys():
            candidates_info[item[4]][item[0]].append(item[2])
        else:
            candidates_info[item[4]][item[0]] = [item[2]]
        if item[0] == item[1]:
            continue
        if item[1] in candidates_info[item[4]].keys():
            candidates_info[item[4]][item[1]].append(item[3])
        else:
            candidates_info[item[4]][item[1]] = [item[3]]
    return candidates_info


def get_scale_factor_for_each_plane(cam, candidates_info, sketch, block):
    """
    介绍：传入camera，对称线的归并组以及对应的block，返回在当前block下三个视角的最佳匹配平面缩放程度
    :param cam: 相机
    :param candidates_info: 对称线情况
    :param sketch: 草图
    :param block: 当前块
    :return: plane的缩放因子[0-1之间]

    """

    intersect_info = sketch.intersect_infor
    # 主平面默认为1
    planes_factor = [[1.0], [], []]
    for plane_dir in range(1, 3):
        temp_scale = []
        for item in intersect_info:
            inter_stroke = item.stroke_id
            # 如果当前处理的stroke不属于当前block，跳过（加速时间）
            if inter_stroke[0] > block[1] or inter_stroke[1] > block[1]:
                continue
            inter_coords = item.inter_coords
            # 从正面看
            if inter_stroke[0] in candidates_info[0].keys() and inter_stroke[1] in candidates_info[plane_dir].keys():
                temp_stroke_0 = inter_stroke[0]
                temp_stroke_1 = inter_stroke[1]
            else:
                continue
            for stroke_info_0 in candidates_info[0][temp_stroke_0]:
                start = stroke_info_0[0]
                end = stroke_info_0[1]
                stroke_dir = end-start
                stroke_dir /= np.linalg.norm(stroke_dir)
                pos_3d = cam.lift_point_close_to_line(inter_coords, start, stroke_dir)
                distance_to_camera_0 = np.linalg.norm(pos_3d - cam.cam_pos)
                for stroke_info_1 in candidates_info[plane_dir][temp_stroke_1]:
                    start = stroke_info_1[0]
                    end = stroke_info_1[1]
                    stroke_dir = end - start
                    stroke_dir /= np.linalg.norm(stroke_dir)
                    pos_3d = cam.lift_point_close_to_line(inter_coords, start, stroke_dir)
                    distance_to_camera_1 = np.linalg.norm(pos_3d - cam.cam_pos)
                    temp_scale.append(distance_to_camera_0/distance_to_camera_1)
        mid_scale = np.median(temp_scale)
        mid_scales = [scale for scale in temp_scale if 0.5*mid_scale < scale < 1.5*mid_scale]
        # 聚类
        cluster = MeanShift(bandwidth=2).fit(np.array(mid_scales).reshape(-1, 1))
        cluster_centers = cluster.cluster_centers_

        cluster_info = np.zeros(len(cluster_centers))
        for item in cluster.labels_:
            cluster_info[item] += 1
        mid_info = np.median(cluster_info)
        # 最终分界线
        all_bound = []
        block_threshold = int(mid_info / 2)
        for cluster_id, cluster_weight in enumerate(cluster_info):
            if cluster_weight > block_threshold:
                all_bound.append(cluster_centers[cluster_id])
        planes_factor[plane_dir] = all_bound
    return planes_factor


def get_candidate_by_stroke(candidate, sketch):
    strokes = sketch.strokes
    stroke_number = len(strokes)
    candidate_of_stroke = [[] for i in range(stroke_number)]
    for index, item in enumerate(candidate):
        candidate_of_stroke[item[0]].append(index)
        if item[0] != item[1]:
            candidate_of_stroke[item[1]].append(index)
    return candidate_of_stroke


def reconstruct_candidate_by_plane_factor(camera, candidates, candidates_of_stroke, sketch, plane_scale_factor, block):
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            continue
        candidates_indexes = candidates_of_stroke[stroke.id]
        for index in candidates_indexes:
            candidate = candidates[index]
            if candidate[0] == stroke.id:
                candidate[2] = camera.cam_pos + plane_scale_factor[candidate[4]] * (
                                                  np.array(candidate[2]) - camera.cam_pos),
            if candidate[1] == stroke.id:
                candidate[3] = camera.cam_pos + plane_scale_factor[candidate[4]] * (
                        np.array(candidate[3]) - camera.cam_pos),


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







