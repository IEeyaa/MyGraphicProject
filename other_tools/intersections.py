import numpy as np
from copy import deepcopy

from other_tools import tools_3d


class IntersectionSimple:
    def __init__(self, inter_id=-1, stroke_ids=None,
                 cam_depths=None, acc_radius=-1, epsilon=0.0, is_fixed=False, fix_depth=-1):
        if cam_depths is None:
            cam_depths = []
        if stroke_ids is None:
            stroke_ids = []
        self.inter_id = inter_id
        self.stroke_ids = stroke_ids
        self.cam_depths = cam_depths
        self.acc_radius = acc_radius
        self.epsilon = epsilon
        self.is_fixed = is_fixed
        self.fix_depth = fix_depth


class LineCoverage:
    def __init__(self, weight, inter_id, stroke_proxy_ids=[], stroke_ids=[]):
        self.weight = weight
        self.inter_id = inter_id
        self.stroke_proxy_ids = stroke_proxy_ids
        self.stroke_ids = stroke_ids


# 每一条线的前后端点
def get_intersection_arc_parameters(sketch):
    # for the line-coverage part of the score function, we need the arc_distance for
    # the two most extreme intersections for each stroke
    extreme_intersections_distances_per_stroke = []
    for s_id in range(len(sketch.strokes)):
        if len(sketch.intersect_dict[s_id]) < 2:
            extreme_intersections_distances_per_stroke.append([0, 1])
            continue
        inter_set = [sketch.intersect_infor[inter_id] for inter_id in sketch.intersect_dict[s_id]]
        arc_params = [inter.inter_params[np.argwhere(np.array(inter.stroke_id) == s_id).flatten()[0]]
                      for inter in inter_set]
        if len(arc_params) < 1:
            extreme_intersections_distances_per_stroke.append([0, 1])
            continue
        extreme_intersections_distances_per_stroke.append([np.min(arc_params), np.max(arc_params)])
    return extreme_intersections_distances_per_stroke


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
        # attribute telling us if one of the strokes is fixed
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

        median_length = max(np.min(lengths[0]), np.min(lengths[1]))
        max_epsilon = 0.1
        inter_simple = IntersectionSimple(inter_id=inter.id,
                                          stroke_ids=inter.stroke_id,
                                          acc_radius=5,
                                          cam_depths=cam_depths,
                                          epsilon=max_epsilon * median_length,
                                          is_fixed=inter.is_fixed,
                                          fix_depth=inter.fix_depth)
        inter_simple.inter_3ds = deepcopy(inter_3ds)
        simple_intersections.append(inter_simple)
    return simple_intersections


def prepare_triple_intersections(sketch):
    per_stroke_triple_intersections = []
    for s_id, s in enumerate(sketch.strokes):
        stroke_dict = {"s_id": s_id,
                       "i_triple_intersections": []}
        for inter_id in sketch.intersect_dict[s_id]:
            inter = sketch.intersect_infor[inter_id]
            i_triple_dict = {"inter_id": inter.id,
                             "k_axes": [[], [], [], [], [], []]}
            if len(inter.adjacent_inter_ids) > 1:
                neigh_inters = [sketch.intersect_infor[inter_id] for inter_id in inter.adjacent_inter_ids]
                for neigh_inter in neigh_inters:
                    if not s_id in neigh_inter.stroke_id:
                        continue
                    i_triple_dict["k_axes"][sketch.strokes[neigh_inter.stroke_id[0]].axis_label].append(neigh_inter.id)
                    i_triple_dict["k_axes"][sketch.strokes[neigh_inter.stroke_id[1]].axis_label].append(neigh_inter.id)
                for axis_label in range(len(i_triple_dict["k_axes"])):
                    i_triple_dict["k_axes"][axis_label] = np.unique(i_triple_dict["k_axes"][axis_label]).tolist()
            if np.sum([len(i_triple_dict["k_axes"][i]) > 0 for i in range(len(i_triple_dict["k_axes"]))]) > 2:
                stroke_dict["i_triple_intersections"].append(i_triple_dict)
        if len(stroke_dict["i_triple_intersections"]) > 0:
            per_stroke_triple_intersections.append(stroke_dict)
    return per_stroke_triple_intersections

