# -*- coding:utf-8 -*-
# @Author: IEeya


# 为ortools作相关的准备，例如线的预处理，以及各种辅助道具的引入
from sketch.symmetric_drive import symmetric_driven_build
from symmetric_build.get_candidate import get_all_candidates_from_sketch
from tools.tools_prepare import get_anchor_from_intersection, get_candidate_by_stroke, get_coverage_params_for_strokes


def prepare_candidates_and_intersections(
        cam,
        sketch,
):
    # 找寻candidate对：
    candidate = get_all_candidates_from_sketch(sketch, cam)
    # 获得所有stroke匹配的candidates序号
    candidates_of_stroke = get_candidate_by_stroke(candidate, sketch)
    # 生成极限点
    coverage_params = get_coverage_params_for_strokes(sketch)
    # 生成所有的锚定情况
    stroke_anchor_info = get_anchor_from_intersection(sketch)

    # 形成block
    # blocks = gather_block_from_symmetric_lines(candidate)
    blocks = [[0, 10], [11, 24], [25, 35], [36, 46]]
    # blocks = [[0, 46]]
    print(blocks)
    answer = symmetric_driven_build(
        sketch=sketch,
        candidate=candidate,
        cam=cam,
        blocks=blocks,
        coverage_params=coverage_params,
        anchor_info=stroke_anchor_info
    )
    return answer
