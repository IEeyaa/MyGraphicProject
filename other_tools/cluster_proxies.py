import numpy as np
from tools import tools_3d


def cluster_lines_non_unique_angles(lines):
    lines = np.array([line for line in lines])
    lines_vec = lines[:, 1] - lines[:, 0]
    for l_id, l in enumerate(lines_vec):
        lines_vec[l_id] /= np.linalg.norm(l)
    if len(lines) == 1:
        return lines, [[0]]
    max_length = 0.1 * np.max(np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1))
    max_angle = 10.0  # 10 degrees
    unique_first_points = np.unique(lines[:, 0], axis=0)
    if unique_first_points.shape[0] == 1:
        return lines, [list(range(len(lines)))]
    cam_ray = unique_first_points[1] - unique_first_points[0]
    cam_ray /= np.linalg.norm(cam_ray)
    projected_lines = np.dot(lines[:, 0], cam_ray)
    t_start = np.min(projected_lines)
    t_end = np.max(projected_lines)
    t_tmp = t_start
    cluster_lists = []
    already_existing_cluster_lists = set()
    angle_origin = np.array([1, 0, 0]).reshape(3, 1)
    angle_start = -180.0
    angle_end = 180.0
    while t_tmp < t_end:
        spatial_line_ids_tmp = np.argwhere(np.logical_and(np.abs(projected_lines - t_tmp) < max_length,
                                                          np.logical_not(
                                                              np.isclose(np.abs(projected_lines - t_tmp) - max_length,
                                                                         0.0)))).flatten().tolist()
        t_tmp += max_length
        if len(spatial_line_ids_tmp) == 0:
            continue
        if len(spatial_line_ids_tmp) == 1:
            if not tuple(spatial_line_ids_tmp) in already_existing_cluster_lists:
                cluster_lists.append(spatial_line_ids_tmp)
                already_existing_cluster_lists.add(tuple(spatial_line_ids_tmp))
            continue
        angle_tmp = angle_start
        while angle_tmp < angle_end:
            angles = np.rad2deg(np.dot(lines_vec[spatial_line_ids_tmp], angle_origin))
            angle_line_ids = np.argwhere(np.logical_and(np.abs(angles - angle_tmp).flatten() < 0.5 * max_angle,
                                                        np.logical_not(np.isclose(
                                                            np.abs(angles - angle_tmp).flatten() - 0.5 * max_angle,
                                                            0.0)))).flatten().tolist()
            if len(angle_line_ids) > 0 and not tuple(
                    np.array(spatial_line_ids_tmp)[angle_line_ids].tolist()) in already_existing_cluster_lists:
                cluster_lists.append(np.array(spatial_line_ids_tmp)[angle_line_ids].tolist())
                already_existing_cluster_lists.add(tuple(np.array(spatial_line_ids_tmp)[angle_line_ids].tolist()))
            angle_tmp += max_angle
    proxies = []
    for cluster in cluster_lists:
        if len(cluster) == 1:
            proxies.append(lines[cluster[0]])
            continue
        proxies.append(np.array([np.mean(lines[cluster, 0], axis=0),
                                 np.mean(lines[cluster, 1], axis=0)]))

    return proxies, cluster_lists


def cluster_lines_non_unique(lines, VERBOSE=False):
    lines = np.array(lines)
    if len(lines) == 1:
        return lines, [[0]]
    max_length = 0.05 * np.max(np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1))
    max_length = 0.10 * np.max(np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1))
    unique_first_points = np.unique(lines[:, 0], axis=0)
    if unique_first_points.shape[0] == 1:
        return lines, [list(range(len(lines)))]
    if np.isclose(max_length, 0.0):
        return lines[0], [list(range(len(lines)))]
    cam_ray = unique_first_points[1] - unique_first_points[0]
    cam_ray /= np.linalg.norm(cam_ray)
    projected_lines = np.dot(lines[:, 0], cam_ray)

    t_start = np.min(projected_lines)
    t_end = np.max(projected_lines)
    t_tmp = t_start
    cluster_lists = []
    already_existing_cluster_lists = set()
    while t_tmp < t_end:
        line_ids_tmp = np.argwhere(np.logical_and(np.abs(projected_lines - t_tmp) < max_length,
                                                  np.logical_not(
                                                      np.isclose(np.abs(projected_lines - t_tmp) - max_length,
                                                                 0.0)))).flatten().tolist()
        t_tmp += max_length
        if len(line_ids_tmp) > 0 and not tuple(line_ids_tmp) in already_existing_cluster_lists:
            cluster_lists.append(line_ids_tmp)
            already_existing_cluster_lists.add(tuple(line_ids_tmp))
    proxies = []
    for cluster in cluster_lists:
        if len(cluster) == 1:
            proxies.append(lines[cluster[0]])
            continue
        proxies.append(np.array([np.mean(lines[cluster, 0], axis=0),
                                 np.mean(lines[cluster, 1], axis=0)]))

    return proxies, cluster_lists


def cluster_proxy_strokes(global_candidate_correspondences, per_stroke_proxies, sketch):
    for s_id in range(len(sketch.strokes)):
        all_candidates = []
        candidate_id_counter = 0
        for corr in global_candidate_correspondences:
            if corr[0] == s_id:
                if tools_3d.line_3d_length(corr[2]) > 0.0:
                    all_candidates.append(corr[2])
                    candidate_id_counter += 1
            if corr[1] == s_id:
                if tools_3d.line_3d_length(corr[3]) > 0.0:
                    all_candidates.append(corr[3])
                    candidate_id_counter += 1

        if len(all_candidates) == 0:
            continue
        if sketch.strokes[s_id].axis_label < 3:
            proxies, per_proxy_line_ids = cluster_lines_non_unique(all_candidates)
        else:  # non axis-aligned stroke
            proxies, per_proxy_line_ids = cluster_lines_non_unique_angles(all_candidates)

        per_stroke_proxies[s_id] = proxies
        cand_proxy_ids = [[] for i in range(candidate_id_counter)]
        for proxy_id, cluster in enumerate(per_proxy_line_ids):
            for l_id in cluster:
                cand_proxy_ids[l_id].append(proxy_id)

        # re affect candidates by proxies
        candidate_id_counter = 0
        for corr_id, corr in enumerate(global_candidate_correspondences):
            if corr[0] == s_id:
                if tools_3d.line_3d_length(corr[2]) > 0.0:
                    global_candidate_correspondences[corr_id][5] = cand_proxy_ids[candidate_id_counter]
                    candidate_id_counter += 1
            if corr[1] == s_id:
                if tools_3d.line_3d_length(corr[3]) > 0.0:
                    global_candidate_correspondences[corr_id][6] = cand_proxy_ids[candidate_id_counter]
                    candidate_id_counter += 1
