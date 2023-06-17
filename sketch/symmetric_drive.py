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
    # 形成cluster:
    stroke_groups = [[] for i in sketch.strokes]
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)
    # 形成block
    blocks = gather_block_from_symmetric_lines(candidate)
    print(blocks)

    # blocks.append([0, len(sketch.strokes)-1])
    # 遍历所有的block，生成结果

    extreme_intersections_distances_per_stroke, stroke_lengths = get_intersection_arc_parameters(sketch)
    per_stroke_triple_intersections = prepare_triple_intersections(sketch)

    final_answer = []
    fixed_strokes = [[] for i in range(0, len(sketch.strokes))]
    fixed_planes_scale_factors = []

    for plane in [0, 1, 2]:
        for block_number in range(0, len(blocks)):

            per_axis_per_stroke_candidate_reconstructions = update_candidate_strokes(
                fixed_strokes, candidate, blocks[block_number], len(sketch.strokes))

            """
                planes_scale_factors
                存储每一个平面的平面缩放系数,传入参数的重点是per_axis_per_stroke_candidate_reconstructions
                针对三个平面，给出所有可能的缩放系数，存储在一个一维数组
                [[0.5, 0.75], [1], [2, 2.25]]代表0号平面的缩放系数可能为0.5, 0.75，其它同理
            """
            plane_scale_factors = get_planes_scale_factors(
                sketch, cam, blocks[block_number], block_number, [0, 1, 2], fixed_strokes,
                fixed_planes_scale_factors, per_axis_per_stroke_candidate_reconstructions)
            """
                将当前main_axis 主轴方向的planes_scale_factors只取第一个即使用定值
            """
            if plane != -1 and block_number > 0:
                plane_scale_factors[plane] = [plane_scale_factors[plane][0]]

            plane_scale_factor_number = [range(len(plane_scale_factor))
                                            for plane_scale_factor in plane_scale_factors]
            """
                planes_combs
                planes_combs将返回所有平面缩放因子可能形成的组合
                加入说[[0.5, 0.75], [1], [1.25, 1.5, 2]], 那么会有
                [0, 0, 0], [0, 0, 1], [0, 0, 2],
                [1, 0, 0], [1, 0, 1], [1, 0, 2]
                六种组合
            """
            planes_combs = list(product(*plane_scale_factor_number))
            best_score = -10000
            tmp_final_answer = []
            best_comb = [0, 0, 0]
            for index_comb, planes_comb in enumerate(planes_combs):
                print("planes_comb: ", str(index_comb) + "/" + str(len(planes_combs)))
                # 重构所有相关的candidates:
                planes_point_normal = []
                refl_mats = []
                for i in range(3):
                    focus_vp = i
                    sym_plane_point = np.zeros(3, dtype=np.float_)
                    sym_plane_normal = np.zeros(3, dtype=np.float_)
                    sym_plane_normal[focus_vp] = 1.0
                    sym_plane_point = cam.cam_pos + plane_scale_factors[i][planes_comb[i]] * (np.array(sym_plane_point) - cam.cam_pos)
                    planes_point_normal.append([sym_plane_point, sym_plane_normal])
                for p, n in planes_point_normal:
                    refl_mat = tools_3d.get_reflection_mat(p, n)
                    refl_mats.append(refl_mat)

                local_planes_scale_factors = [plane_scale_factors[0][planes_comb[0]],
                                              plane_scale_factors[1][planes_comb[1]],
                                              plane_scale_factors[2][planes_comb[2]]]

                local_candidate_correspondences, correspondence_ids = copy_correspondences_batch(
                    candidate, blocks[block_number], fixed_strokes, refl_mats,
                    local_planes_scale_factors,
                    cam, sketch)

                if len(local_candidate_correspondences) == 0:
                    continue

                per_stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
                cluster_proxy_strokes(local_candidate_correspondences,
                                      per_stroke_proxies, sketch)

                intersections_3d_simple = get_intersections_simple_batch(
                    per_stroke_proxies, sketch, cam, blocks[block_number], fixed_strokes)

                line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                                      extreme_intersections_distances_per_stroke)
                tmp_final_answer = []
                score, corr, answer = get_best_candidate_by_score(
                    sketch=sketch,
                    candidates=local_candidate_correspondences,
                    candidates_of_stroke=candidates_of_stroke,
                    per_stroke_triple_intersections=per_stroke_triple_intersections,
                    intersections_3d=intersections_3d_simple,
                    line_coverages=line_coverages_simple,
                    block=blocks[block_number],
                    stroke_anchor_info=stroke_anchor_info,
                    group_infor=per_stroke_proxies,
                    plane_dir=plane,
                    fixed_strokes=fixed_strokes
                )
                if score > best_score:
                    tmp_final_answer = answer
                    best_comb = planes_comb
                    print(best_comb)
            for index, item in enumerate(tmp_final_answer):
                if item is not None:
                    fixed_strokes[index].extend(item)
            fixed_planes_scale_factors.append([plane_scale_factors[0][best_comb[0]],
                                               plane_scale_factors[1][best_comb[1]],
                                               plane_scale_factors[2][best_comb[2]]])
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
    for item in fixed_strokes:
        if item is not None and len(item) > 0:
            final_answer.append(item)
    print(final_answer)
    return final_answer
