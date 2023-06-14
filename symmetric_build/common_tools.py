import numpy as np
from skspatial.objects import Plane
from scipy.spatial import distance
from shapely.geometry import LineString

from sketch.intersections import get_intersections_scale_factors
from tools import tools_3d


class Correspondence:

    def __init__(self, stroke_3ds, stroke_ids, candidate_ids, plane_id,
                 first_inter_stroke_id=-1, snd_inter_stroke_id=-1,
                 masked_correspondences=[]):
        self.stroke_3ds = stroke_3ds
        self.stroke_ids = stroke_ids
        self.candidate_ids = candidate_ids
        self.plane_id = plane_id
        # if self-symmetric, note first and second inter-stroke-ids
        self.first_inter_stroke_id = first_inter_stroke_id
        self.snd_inter_stroke_id = snd_inter_stroke_id
        self.masked_correspondences = masked_correspondences


class Candidate:
    def __init__(self, stroke_3d, stroke_id, plane_id, correspondence_id,
                 candidate_id, first_inter_stroke_id=-1, snd_inter_stroke_id=-1):
        self.stroke_3d = stroke_3d
        self.stroke_id = stroke_id
        self.candidate_id = candidate_id
        self.plane_id = plane_id
        self.correspondence_id = correspondence_id
        # if self-symmetric, note first and second inter-stroke-ids
        self.first_inter_stroke_id = first_inter_stroke_id
        self.snd_inter_stroke_id = snd_inter_stroke_id


def update_candidate_strokes(fixed_strokes, correspondences, batch, nb_strokes):
    per_axis_per_stroke_candidate_reconstructions = [[[] for s_id in range(nb_strokes)]
                                                     for plane_id in range(3)]

    for corr in correspondences:
        if corr[0] > batch[1] or corr[1] > batch[1]:
            continue
        if int(corr[7]) != -1 and corr[7] > batch[1]:
            continue
        if int(corr[8]) != -1 and corr[8] > batch[1]:
            continue
        if len(fixed_strokes[corr[0]]) > 0:
            if not len(per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[0]]) > 0:
                per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[0]] = [fixed_strokes[corr[0]]]
        else:
            per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[0]].append(corr[2])
        if int(corr[0]) == int(corr[1]):
            continue
        if len(fixed_strokes[corr[1]]) > 0:
            if not len(per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[1]]) > 0:
                per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[1]] = [fixed_strokes[corr[1]]]
        else:
            per_axis_per_stroke_candidate_reconstructions[corr[4]][corr[1]].append(corr[3])

    return per_axis_per_stroke_candidate_reconstructions


def equi_resample_polyline(poly, dist):
    dists = np.linalg.norm(poly[1:] - poly[:len(poly) - 1], axis=1)
    poly_len = np.sum(dists)
    if poly_len / dist > 100:
        dist = poly_len / 100
    chord_lengths = np.cumsum(dists) / poly_len
    arc_lengths = np.zeros(len(poly))
    arc_lengths[1:] = chord_lengths
    equi_pts = []
    for t in np.linspace(0, 1, int(poly_len / dist)):
        if np.isclose(t, 0.0):
            equi_pts.append(poly[0])
            continue
        elif np.isclose(t, 1.0):
            equi_pts.append(poly[-1])
            continue
        closest_ids = np.argsort(np.abs(arc_lengths - t))
        min_id = np.min(closest_ids[:2])
        max_id = np.max(closest_ids[:2])
        equi_pts.append(poly[min_id] +
                        (t - arc_lengths[min_id]) / (arc_lengths[max_id] - arc_lengths[min_id]) * \
                        (poly[max_id] - poly[min_id]))
    return np.array(equi_pts)


def get_planes_scale_factors(sketch, camera, batch, batch_id, selected_planes, fixed_strokes,
                             fixed_planes_scale_factors,
                             per_axis_per_stroke_candidate_reconstructions):
    planes_scale_factors = []
    for plane_id in selected_planes:
        if np.sum([len(fixed_strokes[i]) for i in range(len(fixed_strokes))]) == 0:  # batch_id == 0:
            if plane_id == 0:
                planes_scale_factors.append([1.0])
                continue
            tmp_max_scale_factor_modes = 4
            intersections_3d, scale_factors = get_intersections_scale_factors(
                per_axis_per_stroke_candidate_reconstructions, selected_planes[0], plane_id,
                sketch, camera, batch=batch)
            # print(scale_factors)
        # filter intersection outliers
        median_scale_factor = np.median(scale_factors)
        del_scale_factors = [i for i, scale_factor in enumerate(scale_factors)
                             if scale_factor < 0.5 * median_scale_factor or scale_factor > 1.5 * median_scale_factor]
        for i in sorted(del_scale_factors, reverse=True):
            # del intersections_3d[i]
            del scale_factors[i]
        n, bins = np.histogram(scale_factors, bins=40)
        new_scale_factors = []
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            already_picked_inters = set()
            for scale_id in range(len(scale_factors)):
                if scale_factors[scale_id] >= bin_start and scale_factors[scale_id] <= bin_end and \
                        not intersections_3d[scale_id].inter_id in already_picked_inters:
                    already_picked_inters.add(intersections_3d[scale_id].inter_id)
                    new_scale_factors.append(scale_factors[scale_id])
        n, bins = np.histogram(new_scale_factors, bins=bins)

        # print(n)
        # print(bins)
        modes_ranked = np.flip(np.argsort(n))
        scale_factors = [(bins[modes_ranked[i] + 1] + bins[modes_ranked[i]]) / 2.0
                         for i in range(tmp_max_scale_factor_modes)]
        if batch_id > 0 and len(fixed_planes_scale_factors) > 0 and len(fixed_planes_scale_factors[-1]) > 0:
            scale_factors.insert(0, fixed_planes_scale_factors[-1][plane_id])
        planes_scale_factors.append(scale_factors)
    return planes_scale_factors


def get_plane_triplet(cam_pos, x_scale, y_scale, z_scale):
    x_plane_point = np.zeros(3, dtype=np.float)
    x_plane_point = cam_pos + x_scale * (np.array(x_plane_point) - cam_pos)
    y_plane_point = np.zeros(3, dtype=np.float)
    y_plane_point = cam_pos + y_scale * (np.array(y_plane_point) - cam_pos)
    z_plane_point = np.zeros(3, dtype=np.float)
    z_plane_point = cam_pos + z_scale * (np.array(z_plane_point) - cam_pos)
    x_plane = Plane(x_plane_point, np.array([1, 0, 0]))
    y_plane = Plane(y_plane_point, np.array([0, 1, 0]))
    z_plane = Plane(z_plane_point, np.array([0, 0, 1]))
    xyz_inter = z_plane.intersect_line(x_plane.intersect_plane(y_plane))
    return np.array(xyz_inter)


def copy_correspondences_batch(correspondences, batch, fixed_strokes, refl_mats,
                               tmp_planes_scale_factors, camera, sketch,
                               ref_correspondences=None):
    if ref_correspondences is None:
        ref_correspondences = []
    batch_correspondences = []
    batch_correspondence_ids = []
    for corr_id, corr in enumerate(correspondences):
        s_id_0 = corr[0]
        s_id_1 = corr[1]
        if len(ref_correspondences) > 0:
            if not ref_correspondences[corr[4]].has_edge(s_id_0, s_id_1):
                continue
        # old correspondences
        if len(fixed_strokes[s_id_0]) > 0 and len(fixed_strokes[s_id_1]) > 0:
            continue
        # correspondence involving future strokes
        if s_id_0 > batch[1] or s_id_1 > batch[1]:
            continue
        # correspondence involving future strokes
        if corr[7] > batch[1] or corr[8] > batch[1]:
            continue

        if corr[7] != -1 and len(fixed_strokes[corr[7]]) > 0 and \
                corr[8] != -1 and len(fixed_strokes[corr[8]]) > 0:
            first_stroke = np.array(fixed_strokes[corr[7]])
            first_stroke_refl = np.array(tools_3d.apply_hom_transform_to_points(first_stroke, refl_mats[corr[4]]))
            first_len = tools_3d.line_3d_length(first_stroke)
            snd_stroke = np.array(fixed_strokes[corr[8]])
            snd_len = tools_3d.line_3d_length(snd_stroke)
            acc_radius = 0.1 * min(first_len, snd_len)
            dist = np.min(distance.cdist(equi_resample_polyline(first_stroke_refl, acc_radius),
                                         equi_resample_polyline(snd_stroke, acc_radius)))
            if dist < acc_radius:
                batch_correspondences.append([corr[0], corr[1],
                                              camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                      np.array(corr[2]) - camera.cam_pos),
                                              camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                      np.array(corr[3]) - camera.cam_pos),
                                              corr[4], corr[5], corr[6], corr[7], corr[8]])
                batch_correspondence_ids.append(corr_id)
            continue

        # correspondences involving non-fixed strokes
        if s_id_0 <= batch[1] and len(fixed_strokes[s_id_0]) == 0 and \
                s_id_1 <= batch[1] and len(fixed_strokes[s_id_1]) == 0:
            batch_correspondences.append([corr[0], corr[1],
                                          camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                  np.array(corr[2]) - camera.cam_pos),
                                          camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                  np.array(corr[3]) - camera.cam_pos),
                                          corr[4], corr[5], corr[6], corr[7], corr[8]])
            batch_correspondence_ids.append(corr_id)
            continue

        # correspondence involving a fixed stroke and a stroke within batch
        if len(fixed_strokes[s_id_0]) > 0:
            s_0 = np.array(fixed_strokes[s_id_0])
            s_0_refl = tools_3d.apply_hom_transform_to_points(s_0, refl_mats[corr[4]])
            s_0_refl_resampled = equi_resample_polyline(s_0_refl, 0.1 * tools_3d.line_3d_length(s_0_refl))
            s_0_proj = np.array(camera.project_polyline(s_0_refl_resampled))
            s_1 = np.array([p.coords for p in sketch.strokes[s_id_1].points_list])
            # dist = np.min(distance.cdist(s_0_proj, s_1))
            dist = LineString(s_0_proj).distance(LineString(s_1))
            if dist < 2 * max(sketch.strokes[s_id_0].acc_radius,
                              sketch.strokes[s_id_1].acc_radius):
                batch_correspondences.append([corr[0], corr[1],
                                              np.array(fixed_strokes[s_id_0]),
                                              camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                      np.array(corr[3]) - camera.cam_pos),
                                              corr[4], corr[5], corr[6], corr[7], corr[8]])
                batch_correspondence_ids.append(corr_id)
            continue

        if len(fixed_strokes[s_id_1]) > 0:
            s_1 = np.array(fixed_strokes[s_id_1])
            s_1_refl = tools_3d.apply_hom_transform_to_points(s_1, refl_mats[corr[4]])
            s_1_refl_resampled = equi_resample_polyline(s_1_refl, 0.1 * tools_3d.line_3d_length(s_1_refl))
            s_1_proj = np.array(camera.project_polyline(s_1_refl_resampled))
            s_0 = np.array([p.coords for p in sketch.strokes[s_id_0].points_list])
            dist = LineString(s_0).distance(LineString(s_1_proj))
            hauss_dist = LineString(s_0).hausdorff_distance(LineString(s_1_proj))
            if dist < 2 * max(sketch.strokes[s_id_0].acc_radius,
                              sketch.strokes[s_id_1].acc_radius):
                batch_correspondences.append([corr[0], corr[1],
                                              camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                      np.array(corr[2]) - camera.cam_pos),
                                              np.array(fixed_strokes[s_id_1]),
                                              corr[4], corr[5], corr[6], corr[7], corr[8]])
                batch_correspondence_ids.append(corr_id)
            continue

    return batch_correspondences, batch_correspondence_ids


def extract_correspondence_information(input_correspondences, sketch,
                                       acc_radius=0.05,
                                       VERBOSE=False):
    correspondences = []
    per_stroke_candidates = [[] for i in range(len(sketch.strokes))]

    for corr_id, corr in enumerate(input_correspondences):
        stroke_ids = [corr[0], corr[1]]
        plane_id = corr[4]
        candidate_id_0 = len(per_stroke_candidates[stroke_ids[0]])
        candidate_id_1 = len(per_stroke_candidates[stroke_ids[1]])
        strokes_3d = [corr[2], corr[3]]
        masked_correspondences = [[], []]
        inter_stroke_id_0 = corr[7]
        inter_stroke_id_1 = corr[8]
        per_stroke_candidates[stroke_ids[0]].append(
            Candidate(stroke_3d=strokes_3d[0], stroke_id=stroke_ids[0],
                      plane_id=plane_id, correspondence_id=corr_id,
                      first_inter_stroke_id=inter_stroke_id_0,
                      snd_inter_stroke_id=inter_stroke_id_1,
                      candidate_id=candidate_id_0))
        if stroke_ids[0] != stroke_ids[1]:
            per_stroke_candidates[stroke_ids[1]].append(
                Candidate(stroke_3d=strokes_3d[1], stroke_id=stroke_ids[1],
                          plane_id=plane_id, correspondence_id=corr_id,
                          first_inter_stroke_id=inter_stroke_id_0,
                          snd_inter_stroke_id=inter_stroke_id_1,
                          candidate_id=candidate_id_1))
        correspondences.append(
            Correspondence(stroke_3ds=strokes_3d, stroke_ids=stroke_ids,
                           candidate_ids=[candidate_id_0, candidate_id_1],
                           plane_id=plane_id,
                           first_inter_stroke_id=inter_stroke_id_0,
                           snd_inter_stroke_id=inter_stroke_id_1,
                           masked_correspondences=masked_correspondences))
    return correspondences, per_stroke_candidates


def extract_fixed_strokes(batches):
    # 最近的batches中的fixed_strokes列表, 存的是final_proxies的内容
    fixed_strokes = batches[-1]["fixed_strokes"]
    proxies = batches[-1]["final_proxies"]
    for p_id, p in enumerate(proxies):
        if p is not None and len(p) > 0:
            fixed_strokes[p_id] = p
    for i, s in enumerate(fixed_strokes):
        fixed_strokes[i] = np.array(s)
    return fixed_strokes
