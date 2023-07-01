import numpy as np
from copy import deepcopy

from tools import tools_3d
from sketch.sketch_info import LineCoverage, Intersection3D


# candidate lines is a list of ordered tuples of points
def get_line_coverages_simple(intersections_3d, sketch, extreme_distances):
    line_coverages = [[] for i in range(len(sketch.strokes))]

    for vec_id, inter in enumerate(intersections_3d):
        sketch_inter = sketch.intersect_infor[inter.inter_id]
        weight_0 = sketch_inter.inter_params[0]
        weight_1 = sketch_inter.inter_params[1]
        dist_0 = extreme_distances[inter.stroke_ids[0]][1] - extreme_distances[inter.stroke_ids[0]][0]
        dist_1 = extreme_distances[inter.stroke_ids[1]][1] - extreme_distances[inter.stroke_ids[1]][0]
        weight_0 = np.clip(weight_0, extreme_distances[inter.stroke_ids[0]][0],
                           extreme_distances[inter.stroke_ids[0]][1])
        weight_1 = np.clip(weight_1, extreme_distances[inter.stroke_ids[1]][0],
                           extreme_distances[inter.stroke_ids[1]][1])
        weight_0 -= extreme_distances[inter.stroke_ids[0]][0]
        weight_1 -= extreme_distances[inter.stroke_ids[1]][0]

        if dist_0 > 0.0:
            weight_0 /= dist_0
        if dist_1 > 0.0:
            weight_1 /= dist_1

        line_coverages[inter.stroke_ids[0]].append(LineCoverage(
            inter_id=inter.inter_id,
            stroke_ids=inter.stroke_ids,
            weight=weight_0))
        line_coverages[inter.stroke_ids[1]].append(LineCoverage(
            inter_id=inter.inter_id,
            stroke_ids=inter.stroke_ids,
            weight=weight_1))
    return line_coverages


def get_intersections_simple_batch(per_stroke_proxies, sketch, camera,
                                   batch, fixed_strokes):
    total_intersections = 0
    simple_intersections = []
    intersections = sketch.intersect_infor
    for inter in intersections:
        inter.is_fixed = False
        inter.fix_depth = -1
        if len(fixed_strokes[inter.stroke_id[0]]) > 0 and len(fixed_strokes[inter.stroke_id[1]]) > 0:
            continue
        if inter.stroke_id[0] > batch[1] or inter.stroke_id[1] > batch[1]:
            continue
        total_intersections += 1
        cam_depths = [[], []]
        inter_3ds = []
        lengths = [[], []]
        for vec_id, s_id in enumerate(inter.stroke_id):
            if len(fixed_strokes[s_id]) > 0:
                inter.is_fixed = True
                p = np.array(fixed_strokes[s_id])

                line_p = p[0]
                line_v = p[-1] - p[0]
                line_v /= np.linalg.norm(line_v)
                lifted_inter = camera.lift_point_close_to_line(inter.inter_coords, line_p, line_v)

                if lifted_inter is not None:
                    cam_depth = np.linalg.norm(np.array(lifted_inter) - np.array(camera.cam_pos))
                    inter.fix_depth = cam_depth
                    inter_3ds.append(lifted_inter)

                lengths[vec_id].append(tools_3d.line_3d_length(p))
                continue
            for p in per_stroke_proxies[s_id]:
                p = np.array(p)
                # lengths[0] 存储了inter.stroke_id[0]的所有可能长度，lengths[1]同理
                lengths[vec_id].append(tools_3d.line_3d_length(p))
                lifted_inter = camera.lift_point_close_to_polyline_v3(inter.inter_coords, p)

                if lifted_inter is not None:
                    cam_depth = np.linalg.norm(np.array(lifted_inter) - np.array(camera.cam_pos))
                    cam_depths[vec_id].append(cam_depth)
                    inter_3ds.append(lifted_inter)

        if len(cam_depths[0]) == 0 and len(cam_depths[1]) == 0:
            continue
        if len(lengths[0]) == 0 or len(lengths[1]) == 0:
            continue

        median_length = max(np.min(lengths[0]), np.min(lengths[1])) * 0.1
        inter_simple = Intersection3D(inter_id=inter.id,
                                      stroke_ids=inter.stroke_id,
                                      acc_radius=5,
                                      cam_depths=cam_depths,
                                      epsilon=median_length,
                                      is_fixed=inter.is_fixed,
                                      fix_depth=inter.fix_depth)
        inter_simple.inter_3ds = deepcopy(inter_3ds)
        simple_intersections.append(inter_simple)
    return simple_intersections

