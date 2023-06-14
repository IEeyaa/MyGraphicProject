import numpy as np

from itertools import product

from symmetric_build.common_tools import get_planes_scale_factors, update_candidate_strokes, extract_fixed_strokes
from sketch.intersections import prepare_triple_intersections, get_intersection_arc_parameters
from symmetric_build.plane_sweep import plane_sweep
from symmetric_build.select_candidate import gather_construction_from_dif_direction_per_block

selected_planes = [0, 1, 2]


def optimize_symmetry_sketch(sketch, cam, symm_candidates, batches, main_axis=-1, fixed_strokes=[]):
    if len(fixed_strokes) == 0:
        fixed_strokes = [[] for i in range(len(sketch.strokes))]
    fixed_planes_scale_factors = []
    fixed_intersections = []
    fixed_line_coverages = np.zeros(len(sketch.strokes))
    accumulated_obj_value = []
    """
        extreme_intersections_distances_per_stroke
        
        stroke_lengths
        每个笔画三维重建的长度[相对于最长重建笔画长度的比例],[0.5, 0.75, 1]代表0号stroke的长度是最长的0.5倍，以此类推
    """
    extreme_intersections_distances_per_stroke, stroke_lengths = get_intersection_arc_parameters(sketch)
    """
        per_stroke_triple_intersections
        返回所有的笔画和三维平面的相交情况，返回一个数组形式存储的dict
        [
            {
                "s_id": 笔画对应的id
                "i_triple_intersections": 一个数组，存储一个dict
                    [
                        "inter_id": 存储和这个笔画相交的id的编号
                        "k_axes": [[], [], [], []] 分别代表0 1 2 3平面方向(xyz以及非传统方向)有没有对称笔画。
                        只有在2个以上的k_axes中有对称笔画的才会被纳入
                    ]
            },
            {
                ...
            },
        ]
    """
    per_stroke_triple_intersections = prepare_triple_intersections(sketch)
    batches_results = []
    # we optimize iteratively, one optimization per batch
    for batch_id, batch in enumerate(batches):
        """
            per_axis_per_stroke_candidate_reconstructions
            存储每个平面方向上笔画的所有重构可能，存储在一个二维数组
            [0][1]:代表0号平面方向（主平面）上1号笔画的3D重构可能，值是一个数组，每个元素是一个[array(), array()]
            赋值方式:
            per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[0]].append(corr[2])
        """
        per_axis_per_stroke_candidate_reconstructions = update_candidate_strokes(
            fixed_strokes, symm_candidates, batch, len(sketch.strokes))

        """
            planes_scale_factors
            存储每一个平面的平面缩放系数,传入参数的重点是per_axis_per_stroke_candidate_reconstructions
            针对三个平面，给出所有可能的缩放系数，存储在一个一维数组
            [[0.5, 0.75], [1], [2, 2.25]]代表0号平面的缩放系数可能为0.5, 0.75，其它同理
        """
        planes_scale_factors = get_planes_scale_factors(
            sketch, cam, batch, batch_id, selected_planes, fixed_strokes,
            fixed_planes_scale_factors, per_axis_per_stroke_candidate_reconstructions)
        """
            将当前main_axis 主轴方向的planes_scale_factors只取第一个即使用定值
        """
        if main_axis != -1 and batch_id > 0:
            planes_scale_factors[main_axis] = [planes_scale_factors[main_axis][0]]

        planes_scale_factors_numbers = [range(len(planes_scale_factor))
                                        for planes_scale_factor in planes_scale_factors]
        """
            planes_combs
            planes_combs将返回所有平面缩放因子可能形成的组合
            加入说[[0.5, 0.75], [1], [1.25, 1.5, 2]], 那么会有
            [0, 0, 0], [0, 0, 1], [0, 0, 2],
            [1, 0, 0], [1, 0, 1], [1, 0, 2]
            六种组合
        """
        print(planes_scale_factors)
        planes_combs = list(product(*planes_scale_factors_numbers))
        plane_sweep(sketch, cam, batch_id, batch, symm_candidates,
                    fixed_strokes, fixed_intersections, fixed_planes_scale_factors,
                    fixed_line_coverages, planes_scale_factors,
                    planes_combs, main_axis, extreme_intersections_distances_per_stroke,
                    per_stroke_triple_intersections, accumulated_obj_value,
                    batches_results)

    return batches_results, accumulated_obj_value


def optimize_symmetry_sketch_pipeline(sketch, cam, symm_candidates, batches):
    batches_result_axes = []
    obj_value_axes = []

    for tmp_axis in range(3):
        batches_result, obj_value = optimize_symmetry_sketch(sketch, cam,
                                                             symm_candidates, batches,
                                                             main_axis=tmp_axis)
        batches_result_axes.append(batches_result)
        obj_value_axes.append(np.sum(obj_value))
    print(obj_value_axes)
    print("main axis", np.argmax(obj_value_axes))
    batches_result = batches_result_axes[np.argmax(obj_value_axes)]

    batches_result_1, obj_value = optimize_symmetry_sketch(sketch, cam,
                                                           symm_candidates, batches, fixed_strokes=extract_fixed_strokes(batches_result))
    return batches_result, batches_result_1
