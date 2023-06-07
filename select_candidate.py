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
        if not (item[0] >= block[0] and item[1] <= block[1]):
            continue
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


def get_scale_factor_for_each_plane(cam, candidates_info, sketch):
    """

    :param cam: 相机
    :param candidates_info: 对称线情况
    :param sketch: 草图
    :return: plane的缩放因子[0-1之间]

    """

    intersect_info = sketch.intersect_infor.values()
    planes_factor = [1.0, 0.0, 0.0]
    for plane_dir in range(1, 3):
        temp_scale = []
        for item in intersect_info:
            inter_stroke = item[0].stroke_id
            inter_coords = item[0].inter_coords
            # 从正面看
            if inter_stroke[0] in candidates_info[0].keys() and inter_stroke[1] in candidates_info[1].keys():
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
                for stroke_info_1 in candidates_info[1][temp_stroke_1]:
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
        labels, counts = np.unique(cluster.labels_, return_counts=True)
        # 找到具有最多成员的聚类中心
        max_count_index = np.argmax(counts)
        most_frequent_center = cluster_centers[max_count_index]
        planes_factor[plane_dir] = most_frequent_center
    return planes_factor






