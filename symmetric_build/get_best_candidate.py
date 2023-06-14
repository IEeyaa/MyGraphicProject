# -*- coding:utf-8 -*-
# @Author: IEeya

# 该函数主要是一个Integer Program，使用Ortools来求解
import numpy as np
from ortools.linear_solver import pywraplp


def star_selector(array_tuple, tmp_tuple):
    indices = np.array([i for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    values = np.array([v for i, v in enumerate(tmp_tuple) if str(v) != "*"])
    return array_tuple[1][np.all(array_tuple[0][:, indices] == values, axis=1).reshape(-1)]


def get_best_candidate_by_score(
        sketch,
        candidates,
        group_infor,
        candidates_of_group,
        candidates_of_stroke,
        line_coverages,
        block,
        stroke_anchor_info,
        plane_scale_factor,
        plane_dir
):
    stroke_max = block[1] + 1
    plane_max = np.max(np.array([candidate[4] for candidate in candidates])) + 1

    # 使用ortools的linear_solver来解决问题
    symmetric_integer_program = pywraplp.Solver.CreateSolver('SAT_INTEGER_PROGRAMMING')
    if not symmetric_integer_program:
        print("The OR-Tools solver could not be created. Check your installation")
        return
    # 抑制输出
    symmetric_integer_program.SuppressOutput()

    # 定义变量
    s = [symmetric_integer_program.IntVar(0, 1, "s_" + str(item)) for item in range(0, stroke_max)]

    sp = [[symmetric_integer_program.IntVar(0, 1, str(stroke_index) + "_" + str(i)) for i in
           range(0, len(group_infor[stroke_index]))] for stroke_index in range(0, stroke_max)]

    c = [symmetric_integer_program.IntVar(0, 1, "c_" + str(item)) for item in range(0, len(candidates))]

    # 代表这个stroke在这个plane中有候选结果
    stroke_plane_factor = [
        [symmetric_integer_program.IntVar(0, 1, str(stroke_index) + "_" + str(plane_index)) for plane_index in
         range(0, plane_max)] for stroke_index in range(0, stroke_max)]

    # 定义基本约束
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            continue
        symmetric_integer_program.Add(
            sum(sp[stroke.id][index] for index in range(0, len(group_infor[stroke.id]))) <= 1)
        for index in range(0, len(group_infor[stroke.id])):
            symmetric_integer_program.Add(s[stroke.id] >= sp[stroke.id][index])
        for index in range(0, len(group_infor[stroke.id])):
            for ca_index in candidates_of_group[stroke.id][index]:
                symmetric_integer_program.Add(sp[stroke.id][index] >= c[ca_index[0]])

    # 定义平面约束
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            continue
        stroke_candidate = [[index, candidates[index][4]] for index in candidates_of_stroke[stroke.id]]
        all_planes = [item[1] for item in stroke_candidate]
        for plane in [0, 1, 2]:
            if plane in all_planes:
                symmetric_integer_program.Add(stroke_plane_factor[stroke.id][plane] <= sum(c[item[0]]
                                                                                           for item in stroke_candidate
                                                                                           if item[1] == plane))
            else:
                symmetric_integer_program.Add(stroke_plane_factor[stroke.id][plane] == 0)

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
        structured_line_coverage_variables.append([min_vars_array, max_vars_array])
        structured_line_coverage_variables_weights.append([min_vars_weights, max_vars_weights])

    total_coverage = 0
    for s_i in range(len(structured_line_coverage_variables)):
        # Coverage部分（系数为4）
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

    for s_i in range(len(structured_line_coverage_variables)):
        if len(structured_line_coverage_variables[s_i][0]) == 0:
            continue
        # there can at most be one max/min line_coverage
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][0][1]) <= 1)
        symmetric_integer_program.Add(sum(structured_line_coverage_variables[s_i][1][1]) <= 1)

    # symmetric
    total_symmetric = 0
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            break
        if len(candidates_of_stroke[stroke.id]) > 0:
            total_symmetric += sum(stroke_plane_factor[stroke.id])

    # proximity
    total_dist = 0
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            break
        stroke_id = stroke.id
        spks = group_infor[stroke_id]
        for index, item in enumerate(spks):
            candidates_indexes = candidates_of_group[stroke_id][index]
            for candidate_infor in candidates_indexes:
                candidate_index = candidate_infor[0]
                candidate_dist = candidate_infor[1]
                total_dist += c[candidate_index] * candidate_dist

    # anchor
    anchor_info = stroke_anchor_info
    total_anchor = 0
    for stroke in sketch.strokes:
        if stroke.id > block[1]:
            break
        # 代表当前stroke被选中
        total_anchor -= (2 - ((len(anchor_info[stroke.id])) >= 1) - ((len(anchor_info[stroke.id])) >= 2)) * s[stroke.id]

    # coverage

    final_score = total_symmetric * 2 - total_dist * 100 + total_anchor * 5 + total_coverage * 4
    symmetric_integer_program.Maximize(final_score)

    status = symmetric_integer_program.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        result = []
        # 输出各个变量的取值
        print("answer: ", symmetric_integer_program.Objective().Value())
        # for stroke_index in range(stroke_max):
        #     variable_value = s[stroke_index].solution_value()
        #     print(f"Variable s[{stroke_index}]: {variable_value}")
        for stroke_index in range(stroke_max):
            for i in range(len(group_infor[stroke_index])):
                var_value = sp[stroke_index][i].solution_value()
                # print(f"Variable sp[{stroke_index}][{i}]: {var_value}")
                if var_value > 0:
                    result.append(group_infor[stroke_index][i])
        # for stroke_index in range(stroke_max):
        #     for i in range(plane_max):
        #         var_value = stroke_plane_factor[stroke_index][i].solution_value()
        #         print(f"Variable spf[{stroke_index}][{i}]: {var_value}")
        # for candidate_index in range(len(candidates)):
        #     variable_value = c[candidate_index].solution_value()
        #     print(f"Variable c[{candidate_index}]: {variable_value}")
        return result
    else:
        print('求解器未找到最优解。')
