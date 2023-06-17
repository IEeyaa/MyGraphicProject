import numpy as np
from math import acos
from shapely.geometry import Point
from copy import deepcopy

from tools_drawing import bezier, bezier_yu
from tools import tools_3d


class IntersectionSimple:
    def __init__(self, inter_id=-1, stroke_ids=[],
                 cam_depths=[], acc_radius=-1, epsilon=0.0, is_tangential=False,
                 is_fixed=False, fix_depth=-1, is_triplet=False, is_extended=False,
                 is_parallel=False):
        self.inter_id = inter_id
        self.stroke_ids = stroke_ids
        self.cam_depths = cam_depths
        self.acc_radius = acc_radius
        self.epsilon = epsilon
        self.is_tangential = is_tangential
        self.is_fixed = is_fixed
        self.fix_depth = fix_depth
        self.is_triplet = is_triplet
        self.is_extended = is_extended
        self.is_parallel = is_parallel


class Intersection3d:
    def __init__(self, inter_id=-1, inter_3d=[], stroke_ids=[],
                 stroke_candidates=[],
                 candidate_correspondence_ids=[]):
        self.inter_id = inter_id
        self.inter_3d = inter_3d
        self.stroke_ids = stroke_ids
        self.stroke_candidates = stroke_candidates
        self.candidate_correspondence_ids = candidate_correspondence_ids


class LineCoverage:

    def __init__(self, weight, inter_id=-1, stroke_proxy_ids=[], stroke_ids=[]):
        self.weight = weight
        self.inter_id = inter_id
        self.stroke_proxy_ids = stroke_proxy_ids
        self.stroke_ids = stroke_ids

    def __str__(self):
        return "inter_id: " + str(self.inter_id) + ", weight: " + str(self.weight) + ", stroke_ids: " + str(
            self.stroke_ids)


def compute_tangential_intersections(sketch, relax_axis_cstrt=False, VERBOSE=False):
    beziers = [[] for i in range(len(sketch.strokes))]
    for inter in sketch.intersection_graph.get_intersections():
        inter.is_tangential = False
        s_id_0, s_id_1 = inter.stroke_ids
        accept_inter = (sketch.strokes[s_id_0].axis_label < 4 and \
                        sketch.strokes[s_id_0].axis_label == sketch.strokes[s_id_1].axis_label)
        if relax_axis_cstrt:
            accept_inter = (sketch.strokes[s_id_0].axis_label < 4 and sketch.strokes[s_id_1].axis_label < 4)
        if accept_inter:
            intersection_length_0 = inter.inter_params[0][1] - inter.inter_params[0][0]
            intersection_length_1 = inter.inter_params[1][1] - inter.inter_params[1][0]
            if min(intersection_length_0, intersection_length_1) > 0.33:
                inter.is_tangential = True
            continue

        if not (sketch.strokes[s_id_0].axis_label == 5 or sketch.strokes[s_id_1].axis_label == 5):
            continue
        if sketch.strokes[s_id_0].is_ellipse() or sketch.strokes[s_id_1].is_ellipse():
            continue
        first_inter_seg = np.array([sketch.strokes[s_id_0].linestring.eval(t)
                                    for t in np.linspace(inter.inter_params[0][0],
                                                         inter.inter_params[0][1], 10)])
        if sketch.strokes[s_id_0].axis_label == 5:
            closest_t_0 = [bezier_yu.get_closest_t(beziers[s_id_0], inter_p)
                           for inter_p in first_inter_seg]
            tan_0 = np.array([bezier.qprime(beziers[s_id_0], t)
                              for t in closest_t_0])
        else:
            tan_0 = np.repeat(np.array([sketch.strokes[s_id_0].points_list[0].coords - \
                                        sketch.strokes[s_id_0].points_list[-1].coords]), 10,
                              axis=0)
        norm = np.linalg.norm(tan_0, axis=1)
        tan_0[:, 0] /= norm
        tan_0[:, 1] /= norm
        if sketch.strokes[s_id_1].axis_label == 5:
            closest_t_1 = [bezier_yu.get_closest_t(beziers[s_id_1], inter_p)
                           for inter_p in first_inter_seg]
            tan_1 = np.array([bezier.qprime(beziers[s_id_1], t)
                              for t in closest_t_1])
        else:
            tan_1 = np.repeat(np.array([sketch.strokes[s_id_1].points_list[0].coords - \
                                        sketch.strokes[s_id_1].points_list[-1].coords]), 10,
                              axis=0)
        norm = np.linalg.norm(tan_1, axis=1)
        tan_1[:, 0] /= norm
        tan_1[:, 1] /= norm

        tan_angles = [np.rad2deg(acos(min(1.0, np.abs(np.dot(tan_0[i], tan_1[i]))))) for i in range(len(tan_0))]
        median_angle = np.median(tan_angles)

        inter.is_tangential = median_angle < 20.0


def get_likely_intersections(sketch):
    likely_intersections = []
    for inter in sketch.intersection_graph.get_intersections():
        inter.is_triplet = False
        extended_inter = np.any(
            [sketch.strokes[inter.stroke_ids[s_id]].linestring.linestring.distance(Point(inter.inter_coords)) >
             sketch.strokes[inter.stroke_ids[s_id]].acc_radius for s_id in range(2)])
        inter.is_extended = extended_inter
        if inter.inter_id in likely_intersections or len(
                inter.adjacent_inter_ids) < 2:
            continue
        inter.is_triplet = True
        # all outgoing strokes from the intersection "cluster"
        outgoing_strokes = [sketch.strokes[s.stroke_id] for s in
                            sketch.intersection_graph.get_strokes_by_inter_ids(
                                inter.adjacent_inter_ids + [inter.inter_id])]
        # look if they form at least three strokes of different orientations, i.e.
        # they differ more than 5 degree
        if len(outgoing_strokes) < 3:
            continue
        differently_angled_strokes = []
        for s in outgoing_strokes:
            if s.is_curved() or s.axis_label == 5:
                continue
            different_for_all_strokes = True
            for diff_s in differently_angled_strokes:

                if s.get_line_angle(diff_s) < np.deg2rad(5.0) / np.pi:  # 5 degrees
                    different_for_all_strokes = False
                    break
            if different_for_all_strokes:
                differently_angled_strokes.append(s)
        curves_outgoing_strokes = [s.is_curved() or s.axis_label == 5
                                   for s in outgoing_strokes]

        if (np.any(curves_outgoing_strokes) and len(differently_angled_strokes) > 1) or \
                len(differently_angled_strokes) > 2:
            likely_intersections.append(inter.inter_id)

    # we don't throw intersections away if there is no likely intersection at
    # the end of a stroke
    for s_id, s in enumerate(sketch.strokes):
        stroke_inters = sketch.intersection_graph.get_intersections_by_stroke_id(
            s_id)
        if len(stroke_inters) == 0:
            continue
        mid_inter_params = []
        stroke_partners = []
        for inter in stroke_inters:
            curr_s_id = \
                np.argwhere(np.array(inter.stroke_ids)[:] == s_id).flatten()[0]
            stroke_partners.append(inter.stroke_ids[np.argwhere(np.array(inter.stroke_ids)[:] != s_id).flatten()[0]])
            mid_inter_params.append(inter.mid_inter_param[curr_s_id])
        max_inter_param = np.max(mid_inter_params)
        min_inter_param = np.min(mid_inter_params)
        # get all intersections which are within 25% of the endpoints of the stroke
        beginning_inters = []
        end_inters = []
        for inter in stroke_inters:
            curr_s_id = \
                np.argwhere(np.array(inter.stroke_ids)[:] == s_id).flatten()[0]
            if inter.mid_inter_param[curr_s_id] <= min_inter_param + 0.25:
                beginning_inters.append(inter)
            elif inter.mid_inter_param[curr_s_id] >= max_inter_param - 0.25:
                end_inters.append(inter)

        likely_intersections += [inter.inter_id for inter in
                                 beginning_inters]

        likely_intersections += [inter.inter_id for inter in end_inters]

        if s.is_curved() or s.axis_label == 5:
            # don't throw tangential intersections away
            curve_inters = [inter.inter_id for inter in stroke_inters
                            if sketch.strokes[np.array(inter.stroke_ids)[
                    np.array(inter.stroke_ids) != s_id][0]].axis_label == 5]
            likely_intersections += curve_inters

    # keep ellipse intersections
    for inter in sketch.intersection_graph.get_intersections():
        if (sketch.strokes[inter.stroke_ids[0]].axis_label == 5 and \
            sketch.strokes[inter.stroke_ids[0]].is_ellipse()) or \
                (sketch.strokes[inter.stroke_ids[1]].axis_label == 5 and \
                 sketch.strokes[inter.stroke_ids[1]].is_ellipse()):
            likely_intersections.append(inter.inter_id)
    likely_intersections = np.unique(likely_intersections)

    return likely_intersections


def remove_unlikely_intersections(sketch):
    for inter in sketch.intersection_graph.get_intersections():
        inter.is_extended = False

    # prune intersection graph of unlikely intersections
    likely_inter_ids = get_likely_intersections(sketch)
    likely_inter_ids = likely_inter_ids.tolist()

    # get rid of intersections between same axes
    same_axis_inter_ids = [inter.inter_id
                           for inter in sketch.intersection_graph.get_intersections()
                           if sketch.strokes[inter.stroke_ids[0]].axis_label == sketch.strokes[
                               inter.stroke_ids[1]].axis_label
                           and sketch.strokes[inter.stroke_ids[0]].axis_label != 5 and not inter.is_tangential]
    likely_inter_ids += same_axis_inter_ids
    likely_inter_ids += [inter.inter_id for inter in sketch.intersection_graph.get_intersections()
                         if inter.is_tangential]
    likely_inter_ids = np.unique(likely_inter_ids).tolist()
    for del_id in sorted(same_axis_inter_ids, reverse=True):
        if del_id in likely_inter_ids:
            likely_inter_ids.remove(del_id)

    likely_inter_ids = np.array(likely_inter_ids)
    unlikely_inter_ids = [inter.inter_id for inter in sketch.intersection_graph.get_intersections()
                          if not inter.inter_id in likely_inter_ids]
    sketch.intersection_graph.remove_intersections(unlikely_inter_ids)


# 每一条线的前后端点
def get_intersection_arc_parameters(sketch):
    # for the line-coverage part of the score function, we need the arc_distance for
    # the two most extreme intersections for each stroke
    extreme_intersections_distances_per_stroke = []
    stroke_lengths = []
    for s_id in range(len(sketch.strokes)):
        stroke_lengths.append(sketch.strokes[s_id].calculate_length())
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
    stroke_lengths = np.array(stroke_lengths)
    stroke_lengths[:] /= np.max(stroke_lengths)
    return extreme_intersections_distances_per_stroke, stroke_lengths


# candidate lines is a list of ordered tuples of points
def get_line_coverages_simple(intersections_3d, sketch, extreme_distances):
    line_coverages = [[] for i in range(len(sketch.strokes))]

    for vec_id, inter in enumerate(intersections_3d):
        if inter.is_parallel:
            continue
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
        is_parallel = hasattr(inter, "is_parallel") and inter.is_parallel
        if is_parallel:
            median_length = min(np.min(lengths[0]), np.min(lengths[1]))
        inter_simple = IntersectionSimple(inter_id=inter.id,
                                          stroke_ids=inter.stroke_id,
                                          acc_radius=5,
                                          cam_depths=cam_depths,
                                          epsilon=max_epsilon * median_length,
                                          is_tangential=False,
                                          is_fixed=inter.is_fixed,
                                          fix_depth=inter.fix_depth,
                                          is_triplet=False,
                                          is_extended=False,
                                          is_parallel=False)
        inter_simple.inter_3ds = deepcopy(inter_3ds)
        inter_simple.inter_coords = inter.inter_coords
        simple_intersections.append(inter_simple)
    return simple_intersections


def get_intersections_scale_factors(per_axis_per_stroke_candidate_reconstructions,
                                    reference_plane_id, aligned_plane_id, sketch,
                                    camera, batch=[]):
    scale_factors = []
    intersections_3d = []
    for inter in sketch.intersect_infor:
        if len(batch) > 0 and (inter.stroke_id[0] > batch[1] or inter.stroke_id[1] > batch[1]):
            continue
        if (len(per_axis_per_stroke_candidate_reconstructions[reference_plane_id][inter.stroke_id[0]]) > 0 and \
                len(per_axis_per_stroke_candidate_reconstructions[aligned_plane_id][inter.stroke_id[1]]) > 0):
            stroke_id_0 = inter.stroke_id[0]
            stroke_id_1 = inter.stroke_id[1]
        elif (len(per_axis_per_stroke_candidate_reconstructions[reference_plane_id][inter.stroke_id[1]]) > 0 and \
              len(per_axis_per_stroke_candidate_reconstructions[aligned_plane_id][inter.stroke_id[0]]) > 0):
            stroke_id_0 = inter.stroke_id[1]
            stroke_id_1 = inter.stroke_id[0]
        else:
            continue
        for s_0_id, s_0 in enumerate(per_axis_per_stroke_candidate_reconstructions[reference_plane_id][stroke_id_0]):
            if sketch.strokes[stroke_id_0].axis_label == 5:
                if sketch.strokes[stroke_id_0].is_ellipse():
                    inter_3d_0 = camera.lift_point_close_to_polyline_v2(inter.inter_coords, s_0)
                else:
                    inter_3d_0 = camera.lift_point_close_to_polyline(inter.inter_coords, s_0)
            else:
                dir_vec_0 = s_0[-1] - s_0[0]
                dir_vec_0 /= np.linalg.norm(dir_vec_0)
                inter_3d_0 = camera.lift_point_close_to_line(inter.inter_coords, s_0[0], dir_vec_0)
            if inter_3d_0 is None:
                continue
            inter_3d_0_cam_dist = np.linalg.norm(inter_3d_0 - camera.cam_pos)
            for s_1_id, s_1 in enumerate(per_axis_per_stroke_candidate_reconstructions[aligned_plane_id][stroke_id_1]):
                if sketch.strokes[stroke_id_1].axis_label == 5:
                    if sketch.strokes[stroke_id_1].is_ellipse():
                        inter_3d_1 = camera.lift_point_close_to_polyline_v2(inter.inter_coords, s_1)
                    else:
                        inter_3d_1 = camera.lift_point_close_to_polyline(inter.inter_coords, s_1)
                else:
                    dir_vec_1 = s_1[-1] - s_1[0]
                    dir_vec_1 /= np.linalg.norm(dir_vec_1)
                    inter_3d_1 = camera.lift_point_close_to_line(inter.inter_coords, s_1[0], dir_vec_1)
                if inter_3d_1 is None:
                    continue
                inter_3d_1_cam_dist = np.linalg.norm(inter_3d_1 - camera.cam_pos)

                scale_factor = inter_3d_0_cam_dist / inter_3d_1_cam_dist
                scale_factors.append(scale_factor)
                intersections_3d.append(
                    Intersection3d(inter_id=inter.id,
                                   inter_3d=inter_3d_0,
                                   stroke_ids=[stroke_id_0, stroke_id_1],
                                   stroke_candidates=[s_0, s_1]))

    return intersections_3d, scale_factors


# fixed_strokes: array of size len(sketch.strokes)
def get_intersections_scale_factors_fixed_strokes(
        per_axis_per_stroke_candidate_reconstructions, fixed_strokes, batch,
        aligned_plane_id, sketch, camera):
    scale_factors = []
    intersections_3d = []
    for inter in sketch.intersect_infor:
        if inter.stroke_id[0] > batch[1] or inter.stroke_id[1] > batch[1]:
            continue
        stroke_id_0 = -1
        stroke_id_1 = -1
        if (len(fixed_strokes[inter.stroke_id[0]]) > 0 and \
                len(fixed_strokes[inter.stroke_id[1]]) == 0):  # and \
            stroke_id_0 = inter.stroke_id[0]
            stroke_id_1 = inter.stroke_id[1]
        elif (len(fixed_strokes[inter.stroke_id[1]]) > 0 and \
              len(fixed_strokes[inter.stroke_id[0]]) == 0):  # and \
            stroke_id_0 = inter.stroke_id[1]
            stroke_id_1 = inter.stroke_id[0]
        else:
            continue
        s_0 = fixed_strokes[stroke_id_0]
        if sketch.strokes[stroke_id_0].axis_label == 5:
            if sketch.strokes[stroke_id_0].is_ellipse():
                inter_3d_0 = camera.lift_point_close_to_polyline_v2(inter.inter_coords, s_0)
            else:
                inter_3d_0 = camera.lift_point_close_to_polyline(inter.inter_coords, s_0)
        else:
            dir_vec_0 = s_0[-1] - s_0[0]
            if np.isclose(np.linalg.norm(dir_vec_0), 0.0):
                continue
            dir_vec_0 /= np.linalg.norm(dir_vec_0)
            inter_3d_0 = camera.lift_point_close_to_line(inter.inter_coords, s_0[0], dir_vec_0)
        if inter_3d_0 is None:
            # print("is None", inter.inter_coords, s_0[0], dir_vec_0, stroke_id_0, stroke_id_1, inter.inter_id)
            continue
        inter_3d_0_cam_dist = np.linalg.norm(inter_3d_0 - camera.cam_pos)
        for s_1_id, s_1 in enumerate(per_axis_per_stroke_candidate_reconstructions[aligned_plane_id][stroke_id_1]):
            # if stroke_id_1 == 61:
            #    print(aligned_plane_id)
            #    print(per_axis_per_stroke_candidate_reconstructions[aligned_plane_id][stroke_id_1])
            if sketch.strokes[stroke_id_1].axis_label == 5:
                if sketch.strokes[stroke_id_1].is_ellipse():
                    inter_3d_1 = camera.lift_point_close_to_polyline_v2(inter.inter_coords, s_1)
                else:
                    inter_3d_1 = camera.lift_point_close_to_polyline(inter.inter_coords, s_1)
            else:
                dir_vec_1 = s_1[-1] - s_1[0]
                if np.isclose(np.linalg.norm(dir_vec_1), 0.0):
                    continue
                dir_vec_1 /= np.linalg.norm(dir_vec_1)
                inter_3d_1 = camera.lift_point_close_to_line(inter.inter_coords, s_1[0], dir_vec_1)
            if inter_3d_1 is None:
                # print("is None", inter.inter_coords, s_1[0], dir_vec_1, stroke_id_0, stroke_id_1, inter.inter_id)
                continue
            inter_3d_1_cam_dist = np.linalg.norm(inter_3d_1 - camera.cam_pos)

            scale_factor = inter_3d_0_cam_dist / inter_3d_1_cam_dist
            scale_factors.append(scale_factor)
            intersections_3d.append(
                Intersection3d(inter_id=inter.id,
                               inter_3d=inter_3d_0,
                               stroke_ids=[stroke_id_0, stroke_id_1],
                               stroke_candidates=[s_0, s_1]))

    return intersections_3d, scale_factors


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


def relift_intersections(intersections, fixed_strokes, proxies, sketch, cam):
    intersections_3d = []

    for inter in intersections:
        lifted_inter_0 = []
        lifted_inter_1 = []
        sketch_inter = sketch.intersection_graph.get_intersections([inter[2]])[0]
        s_0_id = sketch_inter.stroke_ids[0]
        s_0 = fixed_strokes[s_0_id]
        if proxies[s_0_id] is not None:
            s_0 = proxies[s_0_id]
        if len(s_0) > 0:
            if sketch.strokes[s_0_id].axis_label < 5:
                p = s_0[0]
                vec = s_0[-1] - s_0[0]
                vec /= np.linalg.norm(vec)
                lifted_inter_0 = cam.lift_point_close_to_line(sketch_inter.inter_coords, p, vec)
            else:
                lifted_inter_0 = cam.lift_point_close_to_polyline(sketch_inter.inter_coords, s_0)
            intersections_3d.append([list(lifted_inter_0), int(inter[2])])
        s_1_id = sketch_inter.stroke_ids[1]
        s_1 = fixed_strokes[s_1_id]
        if proxies[s_1_id] is not None:
            s_1 = proxies[s_1_id]
        if len(s_1) > 0:
            if sketch.strokes[s_1_id].axis_label < 5:
                p = s_1[0]
                vec = s_1[-1] - s_1[0]
                vec /= np.linalg.norm(vec)
                lifted_inter_1 = cam.lift_point_close_to_line(sketch_inter.inter_coords, p, vec)
            else:
                lifted_inter_1 = cam.lift_point_close_to_polyline(sketch_inter.inter_coords, s_1)
            intersections_3d.append([list(lifted_inter_1), int(inter[2])])
    return intersections_3d


def get_max_intersection_length(sketch, s_id):
    intersections = sketch.intersection_graph.get_intersections_by_stroke_id(s_id)
    if len(intersections) < 2:
        return sketch.strokes[s_id].linestring.linestring.length
    projected_distances = np.array([sketch.strokes[s_id].linestring.linestring.project(Point(inter.inter_coords))
                                    for inter in intersections])
    max_intersections = np.array([intersections[np.argmin(projected_distances)].inter_coords,
                                  intersections[np.argmax(projected_distances)].inter_coords])

    if len(max_intersections) > 1:
        return np.linalg.norm(max_intersections[0] - max_intersections[-1])
    else:
        return sketch.strokes[s_id].linestring.linestring.length


def get_median_line_lengths(sketch):
    axes_distances = []
    for i in range(3):
        line_lengths = []
        for s_id, s in enumerate(sketch.strokes):
            if s.axis_label != i:
                continue
            line_lengths.append(get_max_intersection_length(sketch, s_id))
        axes_distances.append(np.median(line_lengths))
    return axes_distances
