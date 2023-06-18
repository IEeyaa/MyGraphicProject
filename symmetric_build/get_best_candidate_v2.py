# -*- coding:utf-8 -*-
# @Author: IEeya

# 该函数主要是一个Integer Program，使用Ortools来求解
from copy import deepcopy

import numpy as np
from ortools.linear_solver import pywraplp

# tmp_tuple = (stroke_id, *), array_tuple = (stroke_id, proxy_id) ...
# 将array_tuple中的binary，所有的stroke_id = stroke_id的
from scipy.spatial.distance import directed_hausdorff


def star_selector(array_tuple, tmp_tuple):
    indices = np.array([i for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    values = np.array([v for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    return array_tuple[1][np.all(array_tuple[0][:, indices] == values, axis=1).reshape(-1)]


class Correspondence:

    def __init__(self, stroke_3d, own_stroke_id, own_candidate_id,
                 partner_stroke_id, partner_candidate_id, plane_id, proxy_ids,
                 first_inter_stroke_id=-1, snd_inter_stroke_id=-1,
                 corr_id=-1, proxy_distances=-1):
        self.stroke_3d = stroke_3d
        self.own_stroke_id = own_stroke_id
        self.own_candidate_id = own_candidate_id
        self.partner_stroke_id = partner_stroke_id
        self.partner_candidate_id = partner_candidate_id
        self.plane_id = plane_id
        self.proxy_ids = proxy_ids
        # if self-symmetric, note first and second inter-stroke-ids
        self.first_inter_stroke_id = first_inter_stroke_id
        self.snd_inter_stroke_id = snd_inter_stroke_id
        self.corr_id = corr_id
        self.proxy_distances = proxy_distances


def get_best_candidate_by_score(
        sketch,
        candidates,
        group_infor,
        candidates_of_stroke,
        per_stroke_triple_intersections,
        intersections_3d,
        line_coverages,
        block,
        stroke_anchor_info,
        plane_dir,
        fixed_strokes,
        fixed_intersections,
):
    stroke_max = block[1] + 1
    plane_max = np.max(np.array([candidate[4] for candidate in candidates])) + 1

    per_stroke_per_plane_correspondences = [
        [[] for j in range(plane_max)]
        for i in range(stroke_max)
    ]

    # 创建correspondence
    for corr_id, tmp_corr in enumerate(candidates):
        first_stroke_id = tmp_corr[0]
        snd_stroke_id = tmp_corr[1]
        plane_id = tmp_corr[4]
        first_candidate_id = len(per_stroke_per_plane_correspondences[first_stroke_id][plane_id])
        snd_candidate_id = len(per_stroke_per_plane_correspondences[snd_stroke_id][plane_id])
        first_stroke_3d = tmp_corr[2]
        snd_stroke_3d = tmp_corr[3]
        first_proxy_ids = tmp_corr[5]
        snd_proxy_ids = tmp_corr[6]
        first_inter_stroke_id = tmp_corr[7]
        snd_inter_stroke_id = tmp_corr[8]
        first_proxy_distances = []
        if type(first_proxy_ids) == int:
            continue
        for p_id in first_proxy_ids:
            h_d = max(directed_hausdorff(np.array(first_stroke_3d), np.array(group_infor[first_stroke_id][p_id]))[0],
                      directed_hausdorff(np.array(group_infor[first_stroke_id][p_id]), np.array(first_stroke_3d))[0])
            first_proxy_distances.append(h_d)

        per_stroke_per_plane_correspondences[first_stroke_id][plane_id].append(
            Correspondence(stroke_3d=first_stroke_3d, own_stroke_id=first_stroke_id,
                           own_candidate_id=first_candidate_id, partner_stroke_id=snd_stroke_id,
                           partner_candidate_id=snd_candidate_id, plane_id=plane_id,
                           proxy_ids=first_proxy_ids, first_inter_stroke_id=first_inter_stroke_id,
                           snd_inter_stroke_id=snd_inter_stroke_id, corr_id=corr_id,
                           proxy_distances=deepcopy(first_proxy_distances)))
        if first_stroke_id != snd_stroke_id:
            snd_proxy_distances = []
            for p_id in snd_proxy_ids:
                h_d = max(directed_hausdorff(np.array(snd_stroke_3d), np.array(group_infor[snd_stroke_id][p_id]))[0],
                          directed_hausdorff(np.array(group_infor[snd_stroke_id][p_id]), np.array(snd_stroke_3d))[0])
                snd_proxy_distances.append(h_d)
            per_stroke_per_plane_correspondences[snd_stroke_id][plane_id].append(
                Correspondence(stroke_3d=snd_stroke_3d, own_stroke_id=snd_stroke_id, own_candidate_id=snd_candidate_id,
                               partner_stroke_id=first_stroke_id, partner_candidate_id=first_candidate_id,
                               plane_id=plane_id,
                               proxy_ids=snd_proxy_ids, corr_id=corr_id, proxy_distances=deepcopy(snd_proxy_distances)))

    # 使用ortools的linear_solver来解决问题
    symmetric_integer_program = pywraplp.Solver.CreateSolver('SAT_INTEGER_PROGRAMMING')
    if not symmetric_integer_program:
        print("The OR-Tools solver could not be created. Check your installation")
        return
    # 抑制输出
    symmetric_integer_program.SuppressOutput()

    # 定义变量
    # stroke
    stroke_indices = [s_id for s_id in range(stroke_max)
                      if len(fixed_strokes[s_id]) == 0 and s_id <= block[1]]
    stroke_variables_array = [np.array(stroke_indices).reshape(-1, 1),
                              np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in stroke_indices])]
    stroke_variables_indices_dict = dict([(v, i) for i, v in enumerate(stroke_indices)])

    # (stroke, plane)
    per_stroke_plane_indices = [(s_id, l) for l in range(plane_max)
                                for s_id in range(stroke_max)
                                if len(fixed_strokes[s_id]) == 0 and s_id <= block[1]]
    per_stroke_plane_variables_array = [np.array(per_stroke_plane_indices),
                                        np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in
                                                  per_stroke_plane_indices])]
    per_stroke_plane_indices_dict = dict([(v, i) for i, v in enumerate(per_stroke_plane_indices)])

    # (stroke, proxy)
    proxy_indices = [(s_id, p_id)
                     for s_id in range(stroke_max)
                     for p_id in range(len(group_infor[s_id]))
                     if len(fixed_strokes[s_id]) == 0]

    proxy_variables_array = [np.array(proxy_indices),
                             np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in proxy_indices])]
    proxy_indices_dict = dict([(v, i) for i, v in enumerate(proxy_indices)])

    # correspondence
    correspondence_indices = []
    nb_corr_vars = 0
    nb_corr_links = 0
    cluster_proximity_weights = {}
    for s_id in range(0, block[1] + 1):
        if len(fixed_strokes[s_id]) > 0:
            continue
        for l in range(plane_max):
            for corr in per_stroke_per_plane_correspondences[s_id][l]:
                nb_corr_vars += 1
                for vec_id, p_id in enumerate(corr.proxy_ids):
                    correspondence_indices.append((s_id, p_id, corr.own_candidate_id,
                                                   corr.partner_stroke_id, corr.partner_candidate_id, l,
                                                   corr.corr_id))
                    cluster_proximity_weights[correspondence_indices[-1]] = corr.proxy_distances[vec_id]
                    nb_corr_links += 1
    correspondence_variables_array = [np.array(correspondence_indices),
                                      np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in
                                                correspondence_indices])]

    # intersection
    already_used = set()
    intersection_indices = []
    for inter in intersections_3d:
        inter_name = "i_" + str(inter.stroke_ids[0]) + "_" + str(inter.stroke_ids[1]) + "_" + str(inter.inter_id)
        if inter_name in already_used:
            continue
        already_used.add(inter_name)
        intersection_indices.append((inter.stroke_ids[0], inter.stroke_ids[1], inter.inter_id))

    intersection_variables_array = [np.array(intersection_indices),
                                    np.array(
                                        [symmetric_integer_program.IntVar(0, 1, str(i)) for i in intersection_indices])]
    intersection_indices_dict = dict([(v, i) for i, v in enumerate(intersection_indices)])

    # 定义最终目标规范
    # line coverage
    structured_line_coverage_variables = []
    structured_line_coverage_variables_weights = []
    nb_line_cov = 0
    for s_i in range(len(line_coverages)):
        vars_indices = []
        min_vars_weights = {}
        max_vars_weights = {}
        for j in range(len(line_coverages[s_i])):
            vars_indices.append((s_i, j))
            min_vars_weights[(s_i, j)] = -line_coverages[s_i][j].weight
            max_vars_weights[(s_i, j)] = line_coverages[s_i][j].weight

        min_vars_array = [np.array(vars_indices),
                          np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in vars_indices])]

        max_vars_array = [np.array(vars_indices),
                          np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in vars_indices])]
        nb_line_cov += 2 * len(vars_indices)
        # 针对每一个stroke, 所有可能的coverages
        structured_line_coverage_variables.append([min_vars_array, max_vars_array])
        structured_line_coverage_variables_weights.append([min_vars_weights, max_vars_weights])

    if len(per_stroke_triple_intersections) > 0:
        half_anchored_ids = [s["s_id"] for s in per_stroke_triple_intersections
                             if s["s_id"] in stroke_indices]
        full_anchored_ids = [s["s_id"] for s in per_stroke_triple_intersections
                             if s["s_id"] in stroke_indices]
        half_anchored_vars_array = [np.array(half_anchored_ids).reshape(-1, 1),
                                    np.array(
                                        [symmetric_integer_program.IntVar(0, 1, str(i)) for i in half_anchored_ids])]
        half_anchored_ids_dict = dict([(v, i) for i, v in enumerate(half_anchored_ids)])
        full_anchored_vars_array = [np.array(full_anchored_ids).reshape(-1, 1),
                                    np.array(
                                        [symmetric_integer_program.IntVar(0, 1, str(i)) for i in full_anchored_ids])]
        full_anchored_ids_dict = dict([(v, i) for i, v in enumerate(full_anchored_ids)])

        i_triple_ids = []
        k_axes_ids = []

        potentially_full_anchored_stroke_ids = []
        for s in per_stroke_triple_intersections:
            if not s["s_id"] in stroke_indices:
                continue
            for i_triple_inter in s["i_triple_intersections"]:
                i_triple_ids.append((s["s_id"], i_triple_inter["inter_id"]))
                for i in range(len(i_triple_inter["k_axes"])):
                    k_axes_ids.append((s["s_id"], i_triple_inter["inter_id"], i))

        i_triple_vars_array = [np.array(i_triple_ids),
                               np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in i_triple_ids])]

        max_i_triple_vars_array = [np.array(i_triple_ids),
                                   np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in i_triple_ids])]

        min_i_triple_vars_array = [np.array(i_triple_ids),
                                   np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in i_triple_ids])]
        i_triple_ids_dict = dict([(v, i) for i, v in enumerate(i_triple_ids)])

        k_axes_vars_array = [np.array(k_axes_ids),
                             np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in k_axes_ids])]
        k_axes_ids_dict = dict([(v, i) for i, v in enumerate(k_axes_ids)])

        # constraints
        for s in per_stroke_triple_intersections:
            if not s["s_id"] in stroke_indices:
                continue
            symmetric_integer_program.Add(half_anchored_vars_array[1][half_anchored_ids_dict[s["s_id"]]] <= sum(
                star_selector(i_triple_vars_array, (s["s_id"], "*"))))
            # choose only two i_triple per stroke

            symmetric_integer_program.Add(sum(star_selector(i_triple_vars_array, (s["s_id"], "*"))) <= 2)
            # full anchored strokes must have i_triples which are far away

            # 奇怪的地方？
            ids = [i_triple_inter["inter_id"] for i_triple_inter in s["i_triple_intersections"]]
            tmp_inters = [sketch.intersect_infor[inter_id] for inter_id in ids]
            inter_params = {}
            for tmp_inter in tmp_inters:
                inter_params[tmp_inter.id] = np.array(tmp_inter.inter_params)[
                    np.array(tmp_inter.stroke_id) == s["s_id"]]

                symmetric_integer_program.Add(
                    max_i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] <=
                    i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]])
                symmetric_integer_program.Add(
                    min_i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] <=
                    i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]])
                symmetric_integer_program.Add(
                    i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] <= sum(
                        star_selector(min_i_triple_vars_array, (s["s_id"], "*"))))
                symmetric_integer_program.Add(
                    i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] <= sum(
                        star_selector(max_i_triple_vars_array, (s["s_id"], "*"))))

            if np.max(list(inter_params.values())) - np.min(list(inter_params.values())) >= 0.5:
                potentially_full_anchored_stroke_ids.append(s["s_id"])

            max_t_i_sum = sum([max_i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] *
                               inter_params[tmp_inter.id][0]
                               for i, tmp_inter in enumerate(tmp_inters)])

            min_t_i_sum = sum([min_i_triple_vars_array[1][i_triple_ids_dict[(s["s_id"], tmp_inter.id)]] *
                               inter_params[tmp_inter.id][0]
                               for i, tmp_inter in enumerate(tmp_inters)])

            symmetric_integer_program.Add(max_t_i_sum - min_t_i_sum >= 0.5 - 100.0 * (
                    1 - full_anchored_vars_array[1][full_anchored_ids_dict[s["s_id"]]]))
            symmetric_integer_program.Add(sum(star_selector(max_i_triple_vars_array, (s["s_id"], "*"))) <= 1)
            symmetric_integer_program.Add(sum(star_selector(min_i_triple_vars_array, (s["s_id"], "*"))) <= 1)

            for i_triple_inter in s["i_triple_intersections"]:
                symmetric_integer_program.Add(
                    sum(star_selector(k_axes_vars_array, (s["s_id"], i_triple_inter["inter_id"], "*"))) \
                    >= 3 - 1000.0 * (1 - i_triple_vars_array[1][
                        i_triple_ids_dict[(s["s_id"], i_triple_inter["inter_id"])]]))
                for i in range(len(i_triple_inter["k_axes"])):
                    symmetric_integer_program.Add(
                        k_axes_vars_array[1][k_axes_ids_dict[(s["s_id"], i_triple_inter["inter_id"], i)]] \
                        <= sum(
                            [sum(star_selector(intersection_variables_array, ("*", "*", inter_id))) for inter_id in
                             i_triple_inter["k_axes"][i]]) + np.sum(
                            [inter_id in fixed_intersections for inter_id in i_triple_inter["k_axes"][i]]))

        # obj-term
        half_anchored_mult_vars_array = [np.array(stroke_indices).reshape(-1, 1),
                                         np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in stroke_indices])]
        for s in per_stroke_triple_intersections:
            if not s["s_id"] in stroke_indices:
                continue
            symmetric_integer_program.Add(
                half_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] >= stroke_variables_array[1][
                    stroke_variables_indices_dict[s["s_id"]]] + half_anchored_vars_array[1][
                    half_anchored_ids_dict[s["s_id"]]] - 1)
            symmetric_integer_program.Add(
                half_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] <= stroke_variables_array[1][
                    stroke_variables_indices_dict[s["s_id"]]])
            symmetric_integer_program.Add(half_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] <=
                                          half_anchored_vars_array[1][half_anchored_ids_dict[s["s_id"]]])

        half_anchored_term = -sum([stroke_variables_array[1][stroke_variables_indices_dict[s["s_id"]]] -
                                   half_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]]
                                   for s in per_stroke_triple_intersections if s["s_id"] in stroke_indices])

        full_anchored_mult_vars_array = [np.array(stroke_indices).reshape(-1, 1),
                                         np.array(
                                             [symmetric_integer_program.IntVar(0, 1, str(i)) for i in stroke_indices])]

        for s in per_stroke_triple_intersections:
            if not s["s_id"] in potentially_full_anchored_stroke_ids:
                continue
            symmetric_integer_program.Add(
                full_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] >= stroke_variables_array[1][
                    stroke_variables_indices_dict[s["s_id"]]] + full_anchored_vars_array[1][
                    full_anchored_ids_dict[s["s_id"]]] - 1)
            symmetric_integer_program.Add(
                full_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] <= stroke_variables_array[1][
                    stroke_variables_indices_dict[s["s_id"]]])
            symmetric_integer_program.Add(full_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]] <=
                                          full_anchored_vars_array[1][full_anchored_ids_dict[s["s_id"]]])

        half_anchored_term += -sum([stroke_variables_array[1][stroke_variables_indices_dict[s["s_id"]]] -
                                    full_anchored_mult_vars_array[1][stroke_variables_indices_dict[s["s_id"]]]
                                    for s in per_stroke_triple_intersections if s["s_id"] in stroke_indices
                                    if s["s_id"] in potentially_full_anchored_stroke_ids])

    sym_co = 2.0
    pro_co = -100.0
    anchor_co = 2.0
    cover_co = 4.0

    # symmetric
    obj_expr = 0
    corr_term = 0
    corr_term += sym_co * sum(per_stroke_plane_variables_array[1])
    obj_expr += corr_term

    # proximity
    cluster_proximity_term = 0
    cluster_proximity_term += pro_co * sum(
        [cluster_proximity_weights[v] * correspondence_variables_array[1][i] for i, v in
         enumerate(correspondence_indices)])
    obj_expr += cluster_proximity_term

    # anchor
    # anchor_info = stroke_anchor_info
    # total_anchor = 0
    # for index, stroke_id in enumerate(stroke_variables_array[0]):
    #     if stroke_id[0] > block[1] and len(fixed_strokes[stroke_id[0]]) > 0:
    #         continue
    #     # 代表当前stroke被选中
    #     total_anchor -= (2 - ((len(anchor_info[stroke_id[0]])) >= 1) - ((len(anchor_info[stroke_id[0]])) >= 2)) * \
    #                     stroke_variables_array[1][index]
    # obj_expr += total_anchor * anchor_co

    obj_expr += anchor_co * half_anchored_term

    # coverage
    total_coverage = 0
    for s_i in range(len(structured_line_coverage_variables)):
        # Coverage部分（系数为4）
        if len(fixed_strokes[s_i]) > 0:
            continue

        total_coverage += (sum([sum(star_selector(structured_line_coverage_variables[s_i][0],
                                                  (s_i, i))) *
                                structured_line_coverage_variables_weights[s_i][0][(s_i, i)]
                                for i in range(
                len(structured_line_coverage_variables_weights[s_i][0]))]) +
                           sum([sum(star_selector(structured_line_coverage_variables[s_i][1],
                                                  (s_i, i))) *
                                structured_line_coverage_variables_weights[s_i][1][(s_i, i)]
                                for i in range(
                                   len(structured_line_coverage_variables_weights[s_i][1]))]))

    obj_expr += cover_co * total_coverage

    for s_id in range(stroke_max):
        if len(fixed_strokes[s_id]) == 0:
            symmetric_integer_program.Add(
                sum(star_selector(proxy_variables_array, (s_id, "*"))) <= stroke_variables_array[1][
                    stroke_variables_indices_dict[s_id]])
            # Cap correspondences to one per plane. We only want to know if a certain plane has been chosen
            for l in range(plane_max):
                symmetric_integer_program.Add(
                    sum(star_selector(correspondence_variables_array, (s_id, "*", "*", "*", "*", l, "*"))) +
                    sum(star_selector(correspondence_variables_array,
                                      ("*", "*", "*", s_id, "*", l, "*"))) >= 1 -
                    (1 - per_stroke_plane_variables_array[1][per_stroke_plane_indices_dict[s_id, l]]) * 100.0)

    for s_i in range(len(structured_line_coverage_variables)):
        if len(structured_line_coverage_variables[s_i][0]) == 0:
            continue
        # there can at most be one max/min line_coverage
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][0][1]) <= 1)
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][1][1]) <= 1)

        for j in range(len(line_coverages[s_i])):
            inter_id = line_coverages[s_i][j].inter_id
            symmetric_integer_program.Add(sum(star_selector(structured_line_coverage_variables[s_i][0], ("*", j))) \
                                          <= sum(star_selector(intersection_variables_array, ("*", "*", inter_id))))
            symmetric_integer_program.Add(sum(star_selector(structured_line_coverage_variables[s_i][1], ("*", j))) \
                                          <= sum(star_selector(intersection_variables_array, ("*", "*", inter_id))))
        # there should at least be one max/min line_coverage if there's a 3d intersection
        for inter in intersections_3d:
            if inter.stroke_ids[0] == s_i or inter.stroke_ids[1] == s_i:
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][0][1]))
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][1][1]))

    # the contribution of a single stroke to a proxy can at most be one
    for s_id in range(stroke_max):
        if len(fixed_strokes[s_id]) == 0:
            for p_id in range(len(group_infor[s_id])):
                for other_s_id in range(stroke_max):
                    if s_id != other_s_id:
                        symmetric_integer_program.Add(sum(star_selector(correspondence_variables_array,
                                                                        (
                                                                            s_id, p_id, "*", other_s_id, "*", "*",
                                                                            "*"))) <=
                                                      proxy_variables_array[1][proxy_indices_dict[s_id, p_id]])
                    else:
                        # we want to allow orthogonal AND planar self-symmetric strokes
                        for l in range(plane_max):
                            symmetric_integer_program.Add(sum(star_selector(correspondence_variables_array,
                                                                            (s_id, p_id, "*", other_s_id, "*", l,
                                                                             "*"))) <=
                                                          proxy_variables_array[1][proxy_indices_dict[s_id, p_id]])
                # global symmetry constraint
                if plane_dir > -1:
                    symmetric_integer_program.Add(proxy_variables_array[1][proxy_indices_dict[s_id, p_id]] <= sum(
                        star_selector(correspondence_variables_array, (s_id, p_id, "*", "*", "*", plane_dir, "*"))))

    already_covered_correspondences = set()
    for s_id in range(stroke_max):
        if len(fixed_strokes[s_id]) > 0:
            continue
        for p_id in range(len(group_infor[s_id])):
            # An active proxy must have at least one correspondence
            symmetric_integer_program.Add(proxy_variables_array[1][proxy_indices_dict[s_id, p_id]] <= sum(
                star_selector(correspondence_variables_array, (s_id, p_id, "*", "*", "*", "*", "*"))))

        # coherent symmetry correspondence selection
        for l in range(plane_max):
            for corr in per_stroke_per_plane_correspondences[s_id][l]:
                i_1 = s_id
                k_1 = corr.own_candidate_id
                i_2 = corr.partner_stroke_id
                if len(fixed_strokes[i_2]) > 0:
                    continue
                k_2 = corr.partner_candidate_id
                if i_1 == i_2 and k_1 == k_2:
                    continue
                if (l, i_1, k_1, i_2, k_2) in already_covered_correspondences or \
                        (l, i_2, k_2, i_1, k_1) in already_covered_correspondences:
                    continue
                already_covered_correspondences.add((l, i_1, k_1, i_2, k_2))

                symmetric_integer_program.Add(sum(star_selector(correspondence_variables_array,
                                                                (s_id, "*", "*", corr.partner_stroke_id, "*", l,
                                                                 "*"))) == \
                                              sum(star_selector(correspondence_variables_array,
                                                                (corr.partner_stroke_id, "*", "*", s_id, "*", l, "*"))))

        # coherent self-symmetry selection
        # only select a self-symmetric correspondence if both of their intersecting
        # strokes are symmetric
        # except for ellipses
        for l in range(plane_max):
            for corr in per_stroke_per_plane_correspondences[s_id][l]:
                i_1 = s_id
                i_2 = corr.partner_stroke_id
                k_1 = corr.own_candidate_id
                if i_1 != i_2:
                    continue
                first_inter_stroke_id = corr.first_inter_stroke_id
                snd_inter_stroke_id = corr.snd_inter_stroke_id

                # avoid degenerate cases
                if first_inter_stroke_id == snd_inter_stroke_id:
                    # allow self-symmetric cases where there's no intersection
                    if first_inter_stroke_id == -1:
                        continue
                    symmetric_integer_program.Add(
                        sum(star_selector(correspondence_variables_array, (s_id, "*", k_1, "*", "*", l, "*"))) == 0)
                    continue

                if len(fixed_strokes[first_inter_stroke_id]) > 0 and len(fixed_strokes[snd_inter_stroke_id]) > 0:
                    continue
                if len(fixed_strokes[first_inter_stroke_id]) > 0:
                    tmp = snd_inter_stroke_id
                    snd_inter_stroke_id = first_inter_stroke_id
                    first_inter_stroke_id = tmp

                # get candidate_id for stroke-pair
                stroke_pair_k_1 = -1
                for tmp_k in range(len(per_stroke_per_plane_correspondences[first_inter_stroke_id][l])):
                    if per_stroke_per_plane_correspondences[first_inter_stroke_id][l][
                        tmp_k].partner_stroke_id == snd_inter_stroke_id:
                        stroke_pair_k_1 = tmp_k
                        break
                if stroke_pair_k_1 == -1:
                    # no symmetry correspondence between intersecting strokes
                    symmetric_integer_program.Add(
                        sum(star_selector(correspondence_variables_array, (s_id, "*", k_1, "*", "*", l, "*"))) == 0)
                    continue

                # coherent symmetry selection constraint
                # only allow for this self-symmetry if any symmetry correspondence
                # between the two first strokes has been validated

                symmetric_integer_program.Add(
                    sum(star_selector(correspondence_variables_array, (s_id, "*", k_1, "*", "*", l, "*"))) <= \
                    sum(star_selector(correspondence_variables_array, (
                        first_inter_stroke_id, "*", stroke_pair_k_1, "*", "*", l, "*"))))  # + \

    for vec_id, inter_var in enumerate(intersection_variables_array[0]):
        s_0, s_1, inter_id = inter_var
        epsilon = intersections_3d[vec_id].epsilon
        if intersections_3d[vec_id].is_fixed:
            free_stroke_id = 0
            if len(intersections_3d[vec_id].cam_depths[0]) == 0:
                free_stroke_id = 1
            # avoid empty solutions
            cam_depths_0 = intersections_3d[vec_id].cam_depths[free_stroke_id]
            tmp_proxies = star_selector(proxy_variables_array, (inter_var[free_stroke_id], "*"))
            x_0 = sum([cam_depths_0[i] * tmp_proxies[i]
                       for i in range(len(cam_depths_0))])
            x_1 = intersections_3d[vec_id].fix_depth
            symmetric_integer_program.Add(
                intersection_variables_array[1][intersection_indices_dict[s_0, s_1, inter_id]] <= \
                sum(star_selector(proxy_variables_array, ([s_0, s_1][free_stroke_id], "*"))))
        else:
            cam_depths_0 = intersections_3d[vec_id].cam_depths[0]
            tmp_proxies = star_selector(proxy_variables_array, (s_0, "*"))
            x_0 = sum([cam_depths_0[i] * tmp_proxies[i]
                       for i in range(len(cam_depths_0))])

            cam_depths_1 = intersections_3d[vec_id].cam_depths[1]
            tmp_proxies = star_selector(proxy_variables_array, (s_1, "*"))
            x_1 = sum([cam_depths_1[i] * tmp_proxies[i]
                       for i in range(len(cam_depths_1))])
            symmetric_integer_program.Add(
                intersection_variables_array[1][intersection_indices_dict[s_0, s_1, inter_id]] <= \
                sum(star_selector(proxy_variables_array, (s_0, "*"))))
            symmetric_integer_program.Add(
                intersection_variables_array[1][intersection_indices_dict[s_0, s_1, inter_id]] <= \
                sum(star_selector(proxy_variables_array, (s_1, "*"))))
        symmetric_integer_program.Add(x_0 - x_1 <= epsilon + 100.0 * (
                1 - intersection_variables_array[1][intersection_indices_dict[s_0, s_1, inter_id]]))
        symmetric_integer_program.Add(x_1 - x_0 <= epsilon + 100.0 * (
                1 - intersection_variables_array[1][intersection_indices_dict[s_0, s_1, inter_id]]))

    # for coverage
    for s_i in range(len(structured_line_coverage_variables)):
        if len(structured_line_coverage_variables[s_i][0]) == 0:
            continue
        # there can at most be one max/min line_coverage
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][0][1]) <= 1)
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][1][1]) <= 1)

        for j in range(len(line_coverages[s_i])):
            inter_id = line_coverages[s_i][j].inter_id
            symmetric_integer_program.Add(sum(star_selector(structured_line_coverage_variables[s_i][0], ("*", j))) \
                                          <= sum(star_selector(intersection_variables_array, ("*", "*", inter_id))))
            symmetric_integer_program.Add(sum(star_selector(structured_line_coverage_variables[s_i][1], ("*", j))) \
                                          <= sum(star_selector(intersection_variables_array, ("*", "*", inter_id))))

        # there should at least be one max/min line_coverage if there's a 3d intersection
        for inter in intersections_3d:
            if inter.stroke_ids[0] == s_i or inter.stroke_ids[1] == s_i:
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][0][1]))
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][1][1]))

    symmetric_integer_program.Maximize(obj_expr)

    status = symmetric_integer_program.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        best_obj = symmetric_integer_program.Objective().Value()
        # print("temp score:", best_obj)

        variable_count = 0
        final_correspondences = []
        for corr_id, corr_var in enumerate(correspondence_variables_array[1]):
            if not np.isclose(corr_var.solution_value(), 1.0):
                continue
            corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
            variable_count += 1
            i_1 = corr[0]
            proxy_id = corr[1]
            k_1 = corr[2]
            i_2 = corr[3]
            k_2 = corr[4]
            plane_id = corr[5]

            first_inter_stroke_id = per_stroke_per_plane_correspondences[i_1][plane_id][k_1].first_inter_stroke_id
            snd_inter_stroke_id = per_stroke_per_plane_correspondences[i_1][plane_id][k_1].snd_inter_stroke_id
            final_correspondences.append([i_1, i_2,
                                          per_stroke_per_plane_correspondences[i_1][plane_id][k_1].stroke_3d,
                                          per_stroke_per_plane_correspondences[i_2][plane_id][k_2].stroke_3d,
                                          plane_id, first_inter_stroke_id, snd_inter_stroke_id, proxy_id,
                                          corr[-1]])
        final_proxies = [None] * stroke_max
        final_intersections = []
        final_line_weights = [[-1, -1] for i in range(stroke_max)]
        # TODO: check for conflicting results
        for corr_var in proxy_variables_array[1]:
            if not np.isclose(corr_var.solution_value(), 1.0):
                continue
            corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
            variable_count += 1
            final_proxies[corr[0]] = group_infor[corr[0]][corr[1]]

        for corr_var in intersection_variables_array[1]:
            if not np.isclose(corr_var.solution_value(), 1.0):
                continue
            corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
            variable_count += 1
            final_intersections.append(corr)

        for s_id in range(len(structured_line_coverage_variables)):
            if len(fixed_strokes[s_id]) > 0:
                continue
            for corr_var in structured_line_coverage_variables[s_id][0][1]:
                if not np.isclose(corr_var.solution_value(), 1.0):
                    continue
                variable_count += 1
                corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
                s_i, j = corr
                final_line_weights[s_i][0] = line_coverages[s_i][j].weight
            for corr_var in structured_line_coverage_variables[s_id][1][1]:
                if not np.isclose(corr_var.solution_value(), 1.0):
                    continue
                variable_count += 1
                corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
                s_i, j = corr
                final_line_weights[s_i][1] = line_coverages[s_i][j].weight

        # for index, item in enumerate(final_proxies):
        #     print(index)
        #     if item is None:
        #         continue
        #     for i in item:
        #         print(i)
        return best_obj, final_correspondences, final_proxies, final_intersections
    else:
        print('求解器未找到最优解。')
