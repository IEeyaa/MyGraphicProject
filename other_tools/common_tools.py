import numpy as np
from scipy.spatial import distance
from shapely.geometry import LineString

from other_tools import tools_3d


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
            s_1 = sketch.strokes[s_id_1].lineString
            # dist = np.min(distance.cdist(s_0_proj, s_1))
            dist = LineString(s_0_proj).distance(s_1)
            if dist < 2 * 5:
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
            s_0 = sketch.strokes[s_id_0].lineString
            dist = s_0.distance(LineString(s_1_proj))
            if dist < 2 * 5:
                batch_correspondences.append([corr[0], corr[1],
                                              camera.cam_pos + tmp_planes_scale_factors[corr[4]] * (
                                                      np.array(corr[2]) - camera.cam_pos),
                                              np.array(fixed_strokes[s_id_1]),
                                              corr[4], corr[5], corr[6], corr[7], corr[8]])
                batch_correspondence_ids.append(corr_id)
            continue

    return batch_correspondences, batch_correspondence_ids


