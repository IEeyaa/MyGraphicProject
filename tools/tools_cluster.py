# -*- coding:utf-8 -*-
# @Author: IEeya
import numpy as np
from shapely import LineString
from sklearn import linear_model


def get_connected_sets(adjacency_matrix):
    n = len(adjacency_matrix)
    visited = [False] * n
    connected_sets = []

    for i in range(n):
        if not visited[i]:
            connected_set = []
            dfs(i, adjacency_matrix, visited, connected_set)
            if len(connected_set) > 1:  # 仅当连接的节点数大于1时将其添加到结果中
                connected_sets.append(connected_set)

    return connected_sets


def dfs(node, adjacency_matrix, visited, connected_set):
    visited[node] = True
    connected_set.append(node)

    for neighbor in range(len(adjacency_matrix[node])):
        if adjacency_matrix[node][neighbor] and not visited[neighbor]:
            dfs(neighbor, adjacency_matrix, visited, connected_set)


# 强大的线性拟合
def fit_strokes(strokes):
    regr = linear_model.LinearRegression()

    cluster_points = []
    for c in strokes:
        cluster_points.append(c.lineString.coords[0])
        cluster_points.append(c.lineString.coords[-1])

    cluster_points_coords = np.array([p for p in cluster_points])

    x_train = cluster_points_coords[:, 0].reshape(-1, 1)
    y_train = cluster_points_coords[:, 1].reshape(-1, 1)
    use_x_train = True
    if abs(np.max(cluster_points_coords[:, 1]) - np.min(
            cluster_points_coords[:, 1])) > \
            abs(np.max(cluster_points_coords[:, 0]) - np.min(
                cluster_points_coords[:, 0])):
        use_x_train = False
        x_train = cluster_points_coords[:, 1].reshape(-1, 1)
        y_train = cluster_points_coords[:, 0].reshape(-1, 1)
    regr.fit(x_train, y_train)
    predicted_values = np.array(regr.predict(x_train)).flatten()
    aggr_line = np.array(
        [[cluster_points_coords[p_id, 0], predicted_values[p_id]]
         for p_id in range(len(cluster_points_coords))])
    if not use_x_train:
        aggr_line = np.array(
            [[predicted_values[p_id], cluster_points_coords[p_id, 1]]
             for p_id in range(len(cluster_points_coords))])
    # aggr_line is unsorted. extract extremal points
    x_min = np.min(aggr_line[:, 0])
    x_max = np.max(aggr_line[:, 0])
    y_min = np.min(aggr_line[:, 1])
    y_max = np.max(aggr_line[:, 1])
    min_min = np.array([x_min, y_min])
    min_max = np.array([x_min, y_max])
    max_min = np.array([x_max, y_min])
    max_max = np.array([x_max, y_max])
    dists_min = [np.linalg.norm(p - min_min) for p in aggr_line]
    dists_max = [np.linalg.norm(p - min_max) for p in aggr_line]
    min_id = np.argmin(dists_min)
    max_id = np.argmin(dists_max)
    if dists_min[min_id] < dists_max[max_id]:
        first_point = aggr_line[min_id]
        dists_min = [np.linalg.norm(p - max_max) for p in aggr_line]
        snd_point = aggr_line[np.argmin(dists_min)]
        aggregated_lines = LineString(np.array([first_point, snd_point]))
    else:
        first_point = aggr_line[max_id]
        dists_min = [np.linalg.norm(p - max_min) for p in aggr_line]
        snd_point = aggr_line[np.argmin(dists_min)]
        aggregated_lines = LineString(np.array([first_point, snd_point]))
    return aggregated_lines


# 对3D的所有重建进行聚合，返回各个区域的聚合结果
# candidate[5]要记录的是当前的candidate_id在stroke_id中参与了哪几个编号组的重建
def cluster_3d_lines_correspondence(
    candidates,
    stroke_groups,
    sketch,
):
    # 遍历所有的stroke
    candidates_of_group = [[] for i in range(0, len(sketch.strokes))]

    for stroke in sketch.strokes:
        # 记录重建情况,代表当前stroke所有重建candidate的参与情况
        candidate_belong = [[] for i in candidates]
        stroke_id = stroke.id
        # 获取所有的重建
        stroke_3d_constructions = []
        for candidates_id, item in enumerate(candidates):
            if stroke_id == item[0]:
                stroke_3d_constructions.append([candidates_id, item[2]])
            elif stroke_id == item[1]:
                stroke_3d_constructions.append([candidates_id, item[3]])
        # 对所有的重建进行判断,先定义threshold以及每隔0.1的区间,并且考虑到大部分的线是近似平行的，因此可以直接用起点进行判断
        all_lines = [line[1] for line in stroke_3d_constructions]
        if len(all_lines) == 0:
            continue
        if len(all_lines) == 1:
            stroke_groups[stroke_id] = [stroke_3d_constructions[0][1]]
            candidate_belong[stroke_3d_constructions[0][0]].append(0)
        else:
            points_0 = np.array([line[0] for line in all_lines])
            points_1 = np.array([line[1] for line in all_lines])
            threshold = 0.1 * np.max([np.linalg.norm(points_0[i] - points_1[i]) for i in range(0, len(points_0))])
            all_sp = [line[1][0] for line in stroke_3d_constructions]
            sp_dir = all_sp[1] - all_sp[0]
            sp_dir /= np.linalg.norm(sp_dir)
            sp_projective = np.dot(all_sp, sp_dir)

            # 所有起点中的最近点
            start_sp = np.min(sp_projective, axis=0)
            # 所有起点中的最远点
            end_sp = np.max(sp_projective, axis=0)
            tmp_sp = start_sp
            cluster_line_number = 0
            while tmp_sp < end_sp:
                group_lines = []
                for sp_id, sp in enumerate(sp_projective):
                    if tmp_sp - threshold < sp < tmp_sp + threshold:
                        group_lines.append(stroke_3d_constructions[sp_id][1])
                        candidate_belong[stroke_3d_constructions[sp_id][0]].append(cluster_line_number)

                        if len(candidates_of_group[stroke_id]) <= cluster_line_number:
                            candidates_of_group[stroke_id].append([])
                        candidates_of_group[stroke_id][cluster_line_number].append([stroke_3d_constructions[sp_id][0],
                                                                                    abs(tmp_sp - sp)])
                if len(group_lines) >= 1:
                    points_0 = np.array([line[0] for line in group_lines])
                    points_1 = np.array([line[1] for line in group_lines])
                    cluster_line = [np.mean(points_0, axis=0), np.mean(points_1, axis=0)]
                    # 形成一个新的group
                    stroke_groups[stroke_id].append(cluster_line)
                    cluster_line_number += 1
                tmp_sp += threshold
        # 将信息保存到所有重建中
        for candidates_id, item in enumerate(candidates):
            if stroke_id == item[0]:
                candidates[candidates_id][5] = candidate_belong[candidates_id]
            if stroke_id == item[1]:
                candidates[candidates_id][6] = candidate_belong[candidates_id]
    return candidates_of_group




