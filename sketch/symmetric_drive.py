# -*- coding:utf-8 -*-
# @Author: IEeya

import numpy as np

from other_tools import tools_3d
from other_tools.cluster_proxies import cluster_proxy_strokes
from other_tools.common_tools import copy_correspondences_batch
from other_tools.intersections import get_intersections_simple_batch, get_line_coverages_simple
from symmetric_build.get_best_candidate import get_best_candidate_by_score
from symmetric_build.plane_reconstruct import get_scale_factor_for_each_plane, \
    gather_construction_from_dif_direction_per_block
from tools.tools_cluster import cluster_3d_lines_correspondence


def symmetric_driven_build(
    sketch,
    cam,
    blocks,
    candidate,
    coverage_params,
    anchor_info,
):
    final_answer = []
    final_fixed_strokes = []
    max_score = 0
    for plane in [0, -1]:
        print("checking for plane: ", plane)
        total_score = 0
        fixed_strokes = [[] for i in range(0, len(sketch.strokes))]
        fixed_intersections = []
        for block_number in range(0, len(blocks)):
            print("block: ", block_number)
            candidate_info = gather_construction_from_dif_direction_per_block(candidate, fixed_strokes)
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

            # 源代码函数
            local_candidate_correspondences = copy_correspondences_batch(
                candidate, blocks[block_number], fixed_strokes, refl_mats,
                plane_scale_factor,
                cam, sketch)

            if len(local_candidate_correspondences) == 0:
                continue

            # stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
            # cluster_3d_lines_correspondence(local_candidate_correspondences, stroke_proxies, sketch)
            stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
            cluster_proxy_strokes(local_candidate_correspondences,
                                  stroke_proxies, sketch)
            # 把所有的stroke_proxies 3D化


            intersections_3d_simple = get_intersections_simple_batch(
                stroke_proxies, sketch, cam, blocks[block_number], fixed_strokes)

            line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch, coverage_params)


            score, answer, tmp_fixed_intersections = get_best_candidate_by_score(
                sketch=sketch,
                candidates=local_candidate_correspondences,
                intersections_3d=intersections_3d_simple,
                line_coverages=line_coverages_simple,
                block=blocks[block_number],
                group_infor=stroke_proxies,
                plane_dir=plane,
                fixed_strokes=fixed_strokes,
                anchor_info=anchor_info,
            )

            tmp_final_answer = answer
            total_score += score
            fixed_intersections.extend(tmp_fixed_intersections)
            for index, item in enumerate(tmp_final_answer):
                if item is not None:
                    fixed_strokes[index].extend(item)
        print("score is: ", total_score)
        if total_score > max_score or plane == -1:
            final_fixed_strokes = fixed_strokes
            max_score = total_score

    for index, item in enumerate(final_fixed_strokes):
        if item is not None and len(item) > 0:
            final_answer.append([str(index), item])

    return final_answer
