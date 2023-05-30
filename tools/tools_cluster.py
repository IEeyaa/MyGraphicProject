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
