# -*- coding:utf-8 -*-
# @Author: IEeya

# 该函数主要是一个Integer Program，使用Ortools来求解
from copy import deepcopy

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.spatial.distance import directed_hausdorff


# 返回array_tuple中的与tmp_tuple匹配(其中 * 代表任意值)的二进制值
def star_selector(array_tuple, tmp_tuple):
    indices = np.array([i for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    values = np.array([v for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    return array_tuple[1][np.all(array_tuple[0][:, indices] == values, axis=1).reshape(-1)]


def get_best_candidate_by_score(
        sketch,
        candidates,
        candidate_plane_max,
        group_infor,
        intersections_3d,
        line_coverages,
        block,
        fixed_strokes,
        anchor_info,
):
    stroke_max = block[1] + 1
    plane_max = candidate_plane_max

    # 使用ortools的linear_solver来解决问题
    symmetric_integer_program = pywraplp.Solver.CreateSolver('SAT_INTEGER_PROGRAMMING')
    # 抑制输出
    symmetric_integer_program.SuppressOutput()

    # 定义变量
    stroke_indices = [s_id for s_id in range(stroke_max)
                      if len(fixed_strokes[s_id]) == 0 and s_id <= block[1]]
    stroke_variables_array = [np.array(stroke_indices).reshape(-1, 1),
                              np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in stroke_indices])]
    stroke_variables_indices_dict = dict([(v, i) for i, v in enumerate(stroke_indices)])

    # (stroke, plane, strokes for each plane)
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

    # c_id: 当前的Correspondence在s_id, plane的组合中是第几个（每一个stroke在不同的方向上有不同的重建）
    # 这是一个字典，字典的key是(s_id, p_id, c_id, p_s_id, p_c_id, plane, corr_id)
    cluster_proximity_weights = {}
    for s_id in range(0, block[1] + 1):
        if len(fixed_strokes[s_id]) > 0:
            continue
        for l in range(plane_max):
            for corr in candidates[s_id][l]:
                for vec_id, p_id in enumerate(corr.proxy_info[0]):
                    correspondence_indices.append((s_id, p_id, corr.stroke_indexes[0],
                                                   corr.stroke_2d_ids[1], corr.stroke_indexes[1], l,
                                                   corr.candidate_id))
                    cluster_proximity_weights[correspondence_indices[-1]] = corr.proxy_info[1][vec_id]
    # correspondence 的情况
    correspondence_variables_array = [np.array(correspondence_indices),
                                      np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in
                                                correspondence_indices])]
    # intersection
    intersection_indices = []
    for inter in intersections_3d:
        intersection_indices.append((inter.stroke_ids[0], inter.stroke_ids[1], inter.inter_id))

    intersection_variables_array = [np.array(intersection_indices),
                                    np.array(
                                        [symmetric_integer_program.IntVar(0, 1, str(i)) for i in intersection_indices])]
    intersection_indices_dict = dict([(v, i) for i, v in enumerate(intersection_indices)])

    # 定义最终目标规范
    # 参数
    sym_co = 2.0
    pro_co = -100.0
    anchor_co = 5.0
    cover_co = 10.0

    # symmetric
    obj_expr = 0
    corr_term = 0
    # 显示为每个stroke在每个plane上是否有重建
    corr_term += sym_co * sum(per_stroke_plane_variables_array[1])
    obj_expr += corr_term

    # proximity
    # 遍历所有的correspondence, cluster_proximity_term是每一个correspondence中的proxy_distances
    cluster_proximity_term = 0
    cluster_proximity_term += pro_co * sum(
        [cluster_proximity_weights[v] * correspondence_variables_array[1][i] for i, v in
         enumerate(correspondence_indices)])
    obj_expr += cluster_proximity_term

    # anchor
    total_anchor = 0
    for stroke in sketch.strokes:
        if stroke.id >= block[1] or len(fixed_strokes[stroke.id]) > 0:
            continue
        # 代表当前stroke被选中
        total_anchor -= (2 - ((len(anchor_info[stroke.id])) >= 1) - ((len(anchor_info[stroke.id])) >= 2)) * \
                        sum([stroke_variables_array[1][stroke_variables_indices_dict[stroke.id]]])
    obj_expr += anchor_co * total_anchor

    # line coverage
    structured_line_coverage_variables = []
    structured_line_coverage_variables_weights = []
    for s_i in range(len(line_coverages)):
        vars_indices = []
        vars_weights = {}
        for j in range(len(line_coverages[s_i])):
            vars_indices.append((s_i, j))
            vars_weights[(s_i, j)] = line_coverages[s_i][j].weight
        # 根据论文中的方法，用af和al来代表是否被选中
        min_vars_array = [np.array(vars_indices),
                          np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in vars_indices])]

        max_vars_array = [np.array(vars_indices),
                          np.array([symmetric_integer_program.IntVar(0, 1, str(i)) for i in vars_indices])]
        # 针对每一个stroke, 所有可能的coverages,存储它们的参数
        structured_line_coverage_variables.append([min_vars_array, max_vars_array])
        structured_line_coverage_variables_weights.append(vars_weights)

    total_coverage = 0
    for s_i in range(len(structured_line_coverage_variables)):
        if len(fixed_strokes[s_i]) > 0:
            continue
        for i in range(len(structured_line_coverage_variables_weights[s_i])):
            total_coverage += sum(star_selector(structured_line_coverage_variables[s_i][0], (s_i, i))) * \
                              structured_line_coverage_variables_weights[s_i][(s_i, i)]
            total_coverage -= sum(star_selector(structured_line_coverage_variables[s_i][1], (s_i, i))) * \
                              structured_line_coverage_variables_weights[s_i][(s_i, i)]

    obj_expr += cover_co * total_coverage

    # constrains
    # 一个stroke最多只能有一个proxy
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
                        for l in range(plane_max):
                            symmetric_integer_program.Add(sum(star_selector(correspondence_variables_array,
                                                                            (s_id, p_id, "*", other_s_id, "*", l,
                                                                             "*"))) <=
                                                          proxy_variables_array[1][proxy_indices_dict[s_id, p_id]])

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
            # 获取该平面上所有可能的候选线
            for corr in candidates[s_id][l]:
                i_1 = s_id
                k_1 = corr.stroke_indexes[0]
                i_2 = corr.stroke_2d_ids[1]
                if len(fixed_strokes[i_2]) > 0:
                    continue
                k_2 = corr.stroke_indexes[1]
                if i_1 == i_2 and k_1 == k_2:
                    continue
                if (l, i_1, k_1, i_2, k_2) in already_covered_correspondences or \
                        (l, i_2, k_2, i_1, k_1) in already_covered_correspondences:
                    continue
                already_covered_correspondences.add((l, i_1, k_1, i_2, k_2))

                symmetric_integer_program.Add(sum(star_selector(correspondence_variables_array,
                                                                (s_id, "*", "*", corr.stroke_2d_ids[1], "*", l,
                                                                 "*"))) == \
                                              sum(star_selector(correspondence_variables_array,
                                                                (corr.stroke_2d_ids[1], "*", "*", s_id, "*", l, "*"))))

        # coherent self-symmetry selection
        # only select a self-symmetric correspondence if both of their intersecting
        # strokes are symmetric
        # 约束: 当一个自对称线的对称组都被选中时，选择它
        for l in range(plane_max):
            for corr in candidates[s_id][l]:
                i_1 = s_id
                k_1 = corr.stroke_indexes[0]
                i_2 = corr.stroke_2d_ids[1]
                if i_1 != i_2:
                    continue
                first_inter_stroke_id = corr.stroke_2d_inter_ids[0]
                snd_inter_stroke_id = corr.stroke_2d_inter_ids[1]

                if len(fixed_strokes[first_inter_stroke_id]) > 0 and len(fixed_strokes[snd_inter_stroke_id]) > 0:
                    continue

                if len(fixed_strokes[first_inter_stroke_id]) > 0:
                    tmp = snd_inter_stroke_id
                    snd_inter_stroke_id = first_inter_stroke_id
                    first_inter_stroke_id = tmp

                # get candidate_id for stroke-pair
                stroke_pair_k_1 = -1
                for tmp_k in range(len(candidates[first_inter_stroke_id][l])):
                    if candidates[first_inter_stroke_id][l][tmp_k].stroke_2d_ids[1] == \
                            snd_inter_stroke_id:
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

        # 如果3d重构被使用了，必须选举出min和max的params
        for inter in intersections_3d:
            if inter.stroke_ids[0] == s_i or inter.stroke_ids[1] == s_i:
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][0][1]))
                symmetric_integer_program.Add(
                    sum(star_selector(intersection_variables_array, ("*", "*", inter.inter_id))) <= \
                    sum(structured_line_coverage_variables[s_i][1][1]))

    # final
    symmetric_integer_program.Maximize(obj_expr)

    status = symmetric_integer_program.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        best_obj = symmetric_integer_program.Objective().Value()

        final_proxies = [[] for i in range(stroke_max)]

        for corr_var in proxy_variables_array[1]:
            if not np.isclose(corr_var.solution_value(), 1.0):
                continue
            corr = np.array(corr_var.name().split("(")[1].split(")")[0].split(","), dtype=int).tolist()
            final_proxies[corr[0]] = group_infor[corr[0]][corr[1]]

        return best_obj, final_proxies
    else:
        print('求解器未找到最优解。')
