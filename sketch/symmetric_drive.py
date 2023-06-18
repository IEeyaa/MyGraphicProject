# -*- coding:utf-8 -*-
# @Author: IEeya
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.spatial import distance

from sketch.intersections import get_intersections_simple_batch, get_line_coverages_simple, \
    get_intersection_arc_parameters, prepare_triple_intersections
from sketch.preload_sketch import visualize_lines
from symmetric_build.cluster_proxies import cluster_proxy_strokes
from symmetric_build.common_tools import copy_correspondences_batch, get_planes_scale_factors, update_candidate_strokes
from symmetric_build.gather_block import gather_block_from_symmetric_lines
from symmetric_build.get_best_candidate_v2 import get_best_candidate_by_score
from symmetric_build.get_candidate import get_all_candidates_from_sketch
from symmetric_build.ortools_models import solve_symm_bip_ortools
from symmetric_build.select_candidate import get_scale_factor_for_each_plane, \
    gather_construction_from_dif_direction_per_block, get_anchor_from_intersection, \
    get_candidate_by_stroke, reconstruct_candidate_by_plane_factor
from tools import tools_3d
from tools.tools_cluster import cluster_3d_lines_correspondence


def symmetric_driven_build(cam, sketch):
    block_number = -1
    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 形成cluster:
    stroke_groups = [[] for i in sketch.strokes]
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)
    # 形成block
    blocks = gather_block_from_symmetric_lines(candidate)
    # 遍历所有的block，生成结果
    # for plane_dir in [0, 1, 2]:
    # 针对不同的方向进行预处理
    candidate_info = gather_construction_from_dif_direction_per_block(candidate, blocks[block_number])
    plane_scale_factor = get_scale_factor_for_each_plane(cam, candidate_info, sketch, blocks[block_number])

    # 重构所有相关的candidates:
    reconstruct_candidate_by_plane_factor(cam, candidate, candidates_of_stroke, sketch, plane_scale_factor,
                                          blocks[block_number])
    # 这个至关重要，生成所有组
    candidates_of_group = cluster_3d_lines_correspondence(candidate, stroke_groups, sketch)

    # 3d升空
    fixed = [[] for i in range(len(sketch.strokes))]
    intersections_3d_simple = get_intersections_simple_batch(stroke_groups,
                                                             sketch, cam, blocks[block_number], fixed)
    extreme_intersections_distances_per_stroke, stroke_lengths = get_intersection_arc_parameters(sketch)
    per_stroke_triple_intersections = prepare_triple_intersections(sketch)
    print(per_stroke_triple_intersections)
    exit(1)

    line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                      extreme_intersections_distances_per_stroke)

    score, corr, answer = get_best_candidate_by_score(
        sketch=sketch,
        candidates=candidate,
        candidates_of_stroke=candidates_of_stroke,
        per_stroke_triple_intersections=per_stroke_triple_intersections,
        intersections_3d=intersections_3d_simple,
        line_coverages=line_coverages_simple,
        block=blocks[block_number],
        stroke_anchor_info=stroke_anchor_info,
        group_infor=stroke_groups,
        plane_scale_factor=plane_scale_factor,
        plane_dir=0,
    )

    # result = []
    # for stroke in sketch.strokes:
    #     stroke_id = stroke.id
    #     for item in stroke_groups[stroke_id]:
    #         result.append(item)

    # answer = []
    # stroke_ids = [17, 19, 36]
    # for i in stroke_ids:
    #     stroke_id = i
    #     for item in stroke_groups[stroke_id]:
    #         answer.append(item)

    # visualize_lines(result)
    return answer


def symmetric_driven_build_v2(cam, sketch):
    block_number = 1
    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)
    # 形成block
    blocks = gather_block_from_symmetric_lines(candidate)

    blocks = [[0, 15], [16, 17]]

    print(blocks)

    # blocks.append([0, len(sketch.strokes)-1])
    # 遍历所有的block，生成结果

    extreme_intersections_distances_per_stroke, stroke_lengths = get_intersection_arc_parameters(sketch)
    per_stroke_triple_intersections = prepare_triple_intersections(sketch)

    final_answer = []
    final_fixed_strokes = []
    temp_answer = []
    max_score = 0

    for plane in [0]:
        print("checking for plane: ", plane)
        total_score = 0
        fixed_strokes = [[] for i in range(0, len(sketch.strokes))]
        fixed_intersections = []
        for block_number in range(0, len(blocks)):
            print("block: ", block_number)
            candidate_info = gather_construction_from_dif_direction_per_block(candidate, blocks[block_number], fixed_strokes)
            plane_scale_factor = get_scale_factor_for_each_plane(cam, candidate_info, sketch, blocks[block_number])
            # 重构所有相关的candidates:
            planes_point_normal = []
            refl_mats = []
            for i in range(3):
                focus_vp = i
                sym_plane_point = np.zeros(3, dtype=np.float_)
                sym_plane_normal = np.zeros(3, dtype=np.float_)
                sym_plane_normal[focus_vp] = 1.0
                sym_plane_point = cam.cam_pos + plane_scale_factor[i] * (np.array(sym_plane_point) - cam.cam_pos)
                planes_point_normal.append([sym_plane_point, sym_plane_normal])
            for p, n in planes_point_normal:
                refl_mat = tools_3d.get_reflection_mat(p, n)
                refl_mats.append(refl_mat)

            local_candidate_correspondences, correspondence_ids = copy_correspondences_batch(
                candidate, blocks[block_number], fixed_strokes, refl_mats,
                plane_scale_factor,
                cam, sketch)

            if len(local_candidate_correspondences) == 0:
                continue

            stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
            cluster_proxy_strokes(local_candidate_correspondences,
                                  stroke_proxies, sketch)
            # 把所有的stroke_proxies 3D化
            intersections_3d_simple = get_intersections_simple_batch(
                stroke_proxies, sketch, cam, blocks[block_number], fixed_strokes)
            # if block_number == 1:
            #     for item in intersections_3d_simple:
            #         print(item.stroke_ids, item.inter_id)
            # print(intersections_3d_simple)
            line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                                  extreme_intersections_distances_per_stroke)
            score, corr, answer, tmp_fixed_intersections = get_best_candidate_by_score(
                sketch=sketch,
                candidates=local_candidate_correspondences,
                candidates_of_stroke=candidates_of_stroke,
                per_stroke_triple_intersections=per_stroke_triple_intersections,
                intersections_3d=intersections_3d_simple,
                line_coverages=line_coverages_simple,
                block=blocks[block_number],
                stroke_anchor_info=stroke_anchor_info,
                group_infor=stroke_proxies,
                plane_dir=plane,
                fixed_strokes=fixed_strokes,
                fixed_intersections=fixed_intersections,
            )
            # for item in stroke_proxies[1]:
            #     temp_answer.append(item)

            tmp_final_answer = answer
            total_score += score
            # print(tmp_fixed_intersections)
            fixed_intersections.extend(tmp_fixed_intersections)
            for index, item in enumerate(tmp_final_answer):
                if item is not None:
                    fixed_strokes[index].extend(item)
        print(total_score)
        if total_score > max_score:
            final_fixed_strokes = fixed_strokes
            max_score = total_score

    for index, item in enumerate(final_fixed_strokes):
        if item is not None and len(item) > 0:
            final_answer.append([str(index), item])

    # 根据Intersection_3d拼接剩余的情况

    for index, item in enumerate(temp_answer):
        final_answer.append(["test"+str(index), item])

    return final_answer
