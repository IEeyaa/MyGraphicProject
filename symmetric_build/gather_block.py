# -*- coding:utf-8 -*-
# @Author: IEeya
from math import ceil

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sklearn.cluster import MeanShift

"""
根据对称线聚集形成区块

参数:
    candidates: list -- 对称线的候选列表，每个候选项是一个包含两个整数的列表，表示对称线段的编号，如[4,7]代表4,7号线段形成的对称对。
    min_dist: int -- 最小距离阈值，用于过滤线段长度小于该值的候选项，默认为2（过滤多重笔画）。

返回:
    list -- 最终的区块列表，每个区块是一个包含两个元素的列表，表示区块的起始和结束位置。
"""


def gather_block_from_symmetric_lines(candidates, min_dist=5):
    time_distance = dict()
    """
        形成time_distance_map
        TDM: 用于标识每组线的左右端以及他们的绘制时间跨度。
        例如4: [2, 6]就代表了两条线分别是第2和第6条绘制线，
        且第6条线比第2条线晚4个时间单位绘制
    """
    histogram_length = -1
    for item in candidates:
        left = item[1]
        right = item[0]
        # 更新最大值
        if right > histogram_length:
            histogram_length = right
        # 过滤多重绘画
        if right - left < min_dist:
            continue
        if not right - left in time_distance.keys():
            time_distance[right - left] = [[left, right]]
        else:
            time_distance[right - left].append([left, right])

    histogram = np.zeros(histogram_length)
    candidate_bound = []
    # 形成直方图
    for key in sorted(time_distance)[:3]:
        # 填充bucket
        for item in time_distance[key]:
            histogram[item[0]:item[1] + 1] += 1
        # 将突变的端点纳入
        for i in range(1, len(histogram) - 1):
            if histogram[i] == 0 and histogram[i + 1] > 0 or histogram[i - 1] > 0 and histogram[i] == 0:
                candidate_bound.append(i)
        smoothed_histogram = savgol_filter(histogram, window_length=7, polyorder=3)
        smoothed_histogram[histogram == 0.0] = 0.0
        # 将曲线中的局部最小值纳入
        smallest_index, _ = find_peaks(-smoothed_histogram, prominence=0, distance=1)
        candidate_bound.extend(smallest_index)
    # 聚类, 2个笔画差之内的线聚合在一起
    cluster = MeanShift(bandwidth=2).fit(np.array(candidate_bound).reshape(-1, 1))
    cluster_bound = cluster.cluster_centers_
    cluster_info = np.zeros(len(cluster_bound))
    for item in cluster.labels_:
        cluster_info[item] += 1
    mid_info = np.median(cluster_info)
    # 最终分界线
    all_bound = []
    block_threshold = int(mid_info / 2)
    for cluster_id, cluster_weight in enumerate(cluster_info):
        if cluster_weight > block_threshold:
            all_bound.append(ceil(cluster_bound[cluster_id]))

    all_bound.sort()
    block_info = []
    if not all_bound:
        return [0, histogram_length]

    left_bound = 0
    right_bound = 0
    for item in all_bound:
        right_bound = item
        if right_bound - left_bound >= min_dist:
            block_info.append([left_bound, right_bound])
            left_bound = item+1

    # 处理结尾
    if histogram_length - right_bound > min_dist:
        block_info.append([right_bound, histogram_length])

    # 分解大块
    final_bound = []
    for item in block_info:
        if item[1] - item[0] >= 30:
            half = ceil((item[1] - item[0]) / 2)
            final_bound.append([[item[0], item[0] + half], [item[0] + half + 1, item[1]]])
        else:
            final_bound.append([item[0], item[1]])
    return final_bound
