# -*- coding:utf-8 -*-
# @Author: IEeya

import numpy as np
from sklearn.cluster import MeanShift


def gather_construction_from_dif_direction_per_block(candidates, fixed_strokes):
    """
        :param fixed_strokes: 已经处理过的strokes
        :param candidates: 待筛选的所有候选对称对
        :return: candidates_info:   block中candidates的重建情况
    """
    candidates_info = [{}, {}, {}]
    for item in candidates:
        if item[0] in candidates_info[item[4]].keys():
            if len(fixed_strokes[item[0]]) > 0:
                if not len(candidates_info[item[4]][item[0]]) > 0:
                    candidates_info[item[4]][item[0]] = [fixed_strokes[item[0]]]
            else:
                candidates_info[item[4]][item[0]].append(item[2])
        else:
            candidates_info[item[4]][item[0]] = [item[2]]
        if item[0] == item[1]:
            continue
        if item[1] in candidates_info[item[4]].keys():
            if len(fixed_strokes[item[1]]) > 0:
                if not len(candidates_info[item[4]][item[1]]) > 0:
                    candidates_info[item[4]][item[1]] = [fixed_strokes[item[1]]]
            else:
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





