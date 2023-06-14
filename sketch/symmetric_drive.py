# -*- coding:utf-8 -*-
# @Author: IEeya
from sketch.intersections import get_intersections_simple_batch, get_line_coverages_simple, \
    get_intersection_arc_parameters
from sketch.preload_sketch import visualize_lines
from symmetric_build.gather_block import gather_block_from_symmetric_lines
from symmetric_build.get_best_candidate import get_best_candidate_by_score
from symmetric_build.get_candidate import get_all_candidates_from_sketch
from symmetric_build.select_candidate import get_scale_factor_for_each_plane, \
    gather_construction_from_dif_direction_per_block, get_anchor_from_intersection, \
    get_candidate_by_stroke, reconstruct_candidate_by_plane_factor
from tools.tools_cluster import cluster_3d_lines_correspondence


def symmetric_driven_build(cam, sketch):
    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 形成cluster:
    stroke_groups = [[] for i in sketch.strokes]
    # 这个至关重要，生成所有组
    candidates_of_group = cluster_3d_lines_correspondence(candidate, stroke_groups, sketch)
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)
    # 形成block
    blocks = gather_block_from_symmetric_lines(candidate)
    # 遍历所有的block，生成结果
    # for plane_dir in [0, 1, 2]:
    # 针对不同的方向进行预处理
    candidate_info = gather_construction_from_dif_direction_per_block(candidate, blocks[-1])
    plane_scale_factor = get_scale_factor_for_each_plane(cam, candidate_info, sketch, blocks[-1])

    # 3d升空
    fixed = [[] for i in range(len(sketch.strokes))]
    intersections_3d_simple = get_intersections_simple_batch(stroke_groups,
                                                             sketch, cam, blocks[-1], fixed)
    extreme_intersections_distances_per_stroke, stroke_lengths = get_intersection_arc_parameters(sketch)

    line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                      extreme_intersections_distances_per_stroke)


    # 重构所有相关的candidates:
    reconstruct_candidate_by_plane_factor(cam, candidate, candidates_of_stroke, sketch, plane_scale_factor, blocks[-1])
    result = get_best_candidate_by_score(
        sketch=sketch,
        candidates=candidate,
        candidates_of_group=candidates_of_group,
        candidates_of_stroke=candidates_of_stroke,
        line_coverages=line_coverages_simple,
        block=blocks[-1],
        stroke_anchor_info=stroke_anchor_info,
        group_infor=stroke_groups,
        plane_scale_factor=plane_scale_factor,
        plane_dir=0,
    )
    # result = []
    # for item in candidate:
    #     result.append(item[2])
    #     result.append(item[3])
    print(result)
    visualize_lines(result)
    return blocks
