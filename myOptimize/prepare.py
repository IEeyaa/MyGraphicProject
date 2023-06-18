# -*- coding:utf-8 -*-
# @Author: IEeya


# 为ortools作相关的准备，例如线的预处理，以及各种辅助道具的引入
from other_tools.intersections import get_intersection_arc_parameters, prepare_triple_intersections
from sketch.symmetric_drive import symmetric_driven_build_v2
from symmetric_build.gather_block import gather_block_from_symmetric_lines
from symmetric_build.get_candidate import get_all_candidates_from_sketch
from symmetric_build.select_candidate import get_candidate_by_stroke, get_anchor_from_intersection


def prepare_candidates_and_intersections(
        cam,
        sketch,
):
    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成极限点
    extreme_intersections_distances_per_stroke = get_intersection_arc_parameters(sketch)
    # 组装stroke相交情况
    per_stroke_triple_intersections = prepare_triple_intersections(sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)
    print(stroke_anchor_info)

    # 形成block
    blocks = gather_block_from_symmetric_lines(candidate)
    blocks = [[0, 11], [12, 24], [25, 35], [35, 46]]
    # blocks = [[0, 46]]
    print(blocks)
    answer = symmetric_driven_build_v2(
        sketch=sketch,
        candidate=candidate,
        cam=cam,
        blocks=blocks,
        extreme_intersections_distances_per_stroke=extreme_intersections_distances_per_stroke,
        per_stroke_triple_intersections=per_stroke_triple_intersections,
        anchor_info=stroke_anchor_info
    )
    return answer
