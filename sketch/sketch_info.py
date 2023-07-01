# -*- coding:utf-8 -*-
# @Author: IEeya
import math

import numpy as np
from shapely import Point, STRtree

from pylowstroke.sketch_camera import estimate_initial_camera_parameters
from pylowstroke.sketch_vanishing_points import get_vanishing_points
from tools.tools_2d import get_intersection_info
from tools.tools_cluster import get_connected_sets, fit_strokes


class Stroke:
    def __init__(self, line_string):
        """
        初始化Stroke对象。

        参数：
        - stroke_points：表示折线的点坐标列表

        Stroke对象包含以下属性：
        - pointNumber：折线的点数量
        - coordinates：折线的点坐标列表
        - length：折线的长度

        """
        self.id = -1
        self.lineString = line_string
        self.pointNumber = len(line_string.coords)
        self.length = self.calculate_length()
        self.axis_label = -1
        self.is_curved = False

    def set_stroke_id(self, stroke_id):
        self.id = stroke_id

    def calculate_length(self):
        """
        计算折线的长度。

        返回值：
        - length：折线的长度

        """
        return self.lineString.length

    def intersects(self, other_stroke):
        """
        判断当前的Stroke对象是否与另一个Stroke对象相交。

        参数：
        - other_stroke：另一个Stroke对象

        返回值：
        - is_intersect：布尔值，表示两个Stroke对象是否相交

        """
        line1 = self.lineString
        line2 = other_stroke.lineString
        is_intersect = line1.intersects(line2)
        return is_intersect

    def calculate_angle_difference(self, other_stroke):
        """
        计算与另一个Stroke对象之间的角度差。

        参数：
        - other_stroke：另一个Stroke对象

        返回：
        - angle_difference：角度差（以度数表示）
        """
        vector1 = self.lineString.coords[-1][0] - self.lineString.coords[0][0], self.lineString.coords[-1][1] - \
                  self.lineString.coords[0][1]
        vector2 = other_stroke.lineString.coords[-1][0] - other_stroke.lineString.coords[0][0], \
                  other_stroke.lineString.coords[-1][1] - other_stroke.lineString.coords[0][1]

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude_product = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2) * math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # 检查向量的长度是否接近零
        if np.isclose(magnitude_product, 0):
            angle_difference_degrees = 0
        else:
            # 计算角度差
            angle_difference = math.acos(max(min(dot_product / magnitude_product, 1.0), -1.0))
            angle_difference_degrees = math.degrees(angle_difference)

        return angle_difference_degrees

    def calculate_dis_different(self, stroke):
        return self.lineString.distance(stroke.lineString)


class Sketch:
    def __init__(self, width, height, pen_width, strokes=None):
        self.width = width
        self.height = height
        self.pen_width = pen_width
        if strokes is None:
            self.strokes = []
        else:
            self.strokes = strokes

        # 相交矩阵
        self.intersect_map = np.zeros([len(strokes), len(strokes)], dtype=np.bool_)
        # 相交交叉表
        self.intersect_dict = dict()
        # 相交列表
        self.intersect_infor = []
        # 相交邻接表
        self.intersect_neighbor = dict()
        # 集群列表
        self.line_cluster_list = None
        self.update_stroke_index()

    def filter_strokes(self, threshold=None):
        """
        过滤长度过短的线段。

        参数：
        - strokes：包含多个Stroke对象的列表

        """
        if threshold is None:
            total_length = sum(stroke.length for stroke in self.strokes)
            average_length = total_length / len(self.strokes)
            threshold = average_length / 5.0

        for stroke in reversed(self.strokes):
            if stroke.length <= threshold:
                self.strokes.remove(stroke)

        self.update_stroke_index()

    def update_stroke_index(self):
        for index, stroke in enumerate(self.strokes):
            stroke.set_stroke_id(index)

    def get_intersect_map(self):
        self.intersect_map = np.zeros([len(self.strokes), len(self.strokes)], dtype=np.bool_)
        for stroke in self.strokes:
            intersects_id = [inter.id for inter in self.strokes
                             if stroke.intersects(inter)]
            self.intersect_map[stroke.id, intersects_id] = True
            self.intersect_map[intersects_id, stroke.id] = True
            self.intersect_map[stroke.id, stroke.id] = False

    def get_intersect_info(self):
        # 生成所有的交叉情况
        index = 0
        for stroke in self.strokes:
            self.intersect_dict[stroke.id] = []
        for stroke in self.strokes:
            intersect_stroke_ids = self.get_stroke_neighborhood(stroke.id)
            # 得到所有的intersect_middle_params情况
            for intersect_stroke_id in intersect_stroke_ids:
                intersect_stroke = self.strokes[intersect_stroke_id]
                intersect_params_info = get_intersection_info(stroke.lineString, intersect_stroke.lineString)
                self.intersect_infor.append((Intersection(index, [stroke.id, intersect_stroke_id],
                                                          intersect_params_info[0],
                                                          intersect_params_info[1])))
                self.intersect_dict[stroke.id].append(index)
                index += 1

    def get_adjacent_intersections(self):
        intersections = self.intersect_infor
        points = [Point(inter.inter_coords) for inter in intersections]
        points_inter_ids = [inter.id for inter in intersections]
        index_by_id = {i: points_inter_ids[i] for i, _ in enumerate(intersections)}
        tree = STRtree(points)

        for inter in intersections:
            inter_id = inter.id
            query_geom = points[inter_id].buffer(10)
            inter_neighbours = [inter_p for inter_p in tree.query(query_geom)
                                if query_geom.intersects(points[inter_p]) and
                                np.sum(np.in1d(intersections[index_by_id[inter_p]].stroke_id,
                                               inter.stroke_id)) > 0]
            self.intersect_infor[inter_id].adjacent_inter_ids = inter_neighbours

    def get_stroke_neighborhood(self, stroke_id):
        return [index for index, flag in enumerate(self.intersect_map[stroke_id]) if flag]

    def delete_stroke_by_index(self, indexes):
        for index in sorted(indexes, reverse=True):
            del self.strokes[index]
        self.update_stroke_index()

    def judge_lines(self, threshold_distance):
        for stroke in self.strokes:
            # 将LineString的坐标提取为NumPy数组形式
            line_string = stroke.lineString
            coordinates = np.array(line_string.coords)

            # 使用最小二乘法拟合直线
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            if abs(c) < 600:
                # 计算拟合直线上每个点到实际线上的垂直距离
                distances = np.abs(m * x + c - y) / np.sqrt(m ** 2 + 1)
                # 检查距离是否小于阈值，判断为直线
                if np.mean(distances) >= threshold_distance:
                    stroke.is_curved = True
                else:
                    stroke.is_curved = False
                    # self.strokes[stroke.id].lineString = LineString([(x[0], y[0]), (x[-1], y[-1])])
            else:
                stroke.is_curved = False
                # self.strokes[stroke.id].lineString = LineString([(x[0], y[0]), (x[-1], y[-1])])

    def get_line_cluster(self, threshold_distance, threshold_angle, threshold_index):
        line_cluster_map = np.zeros([len(self.strokes), len(self.strokes)], dtype=np.bool_)
        line_cluster_map[:, :] = False
        intersect_map = self.intersect_map
        for stroke in self.strokes:
            # 首先是直线
            if not stroke.is_curved:
                stroke_id = stroke.id
                # 找出所有存在交叉情况的线（这里有疑义，我觉得应该是距离小于一个数的）
                inters = [inter_stroke.id for inter_stroke in self.strokes if
                          (intersect_map[stroke_id][inter_stroke.id] or
                           stroke.lineString.distance(inter_stroke.lineString) <= threshold_distance
                           and inter_stroke.id != stroke_id)]
                # 判断这些线的情况
                for inter_id in inters:
                    if line_cluster_map[stroke_id, inter_id]:
                        continue
                    if stroke.calculate_angle_difference(self.strokes[inter_id]) <= threshold_angle \
                            and abs(inter_id - stroke_id) <= threshold_index:
                        line_cluster_map[stroke_id, inter_id] = True
                        line_cluster_map[inter_id, stroke_id] = True

        temp_cluster_list = get_connected_sets(line_cluster_map)
        self.line_cluster_list = temp_cluster_list

    def generate_line_from_cluster(self):
        # 遍历所有的line_cluster
        for item in reversed(self.line_cluster_list):
            # 获取所有stroke
            strokes = [self.strokes[index] for index in item]
            # 合并
            cluster_line_stroke_lineString = fit_strokes(strokes)
            # 删除其它笔画
            self.delete_stroke_by_index(item[1:])
            self.strokes[item[0]].lineString = cluster_line_stroke_lineString
        self.update_stroke_index()

    def get_vanishing_points(self):
        vps, p, failed = get_vanishing_points(self)
        return vps, p, failed

    def estimate_camera(self):
        vps, p, failed = self.get_vanishing_points()
        if failed:
            print("camera error")
            exit(1)
        cam_param, lines_group, vps, vp_new_ind = estimate_initial_camera_parameters(vps, p, self)
        return cam_param, lines_group


class Intersection:
    def __init__(self, index, stroke_id, inter_coords, inter_params):
        """
        交叉对象的存储。

        参数：
        - stroke_points：表示折线的点坐标列表

        Stroke对象包含以下属性：
        - pointNumber：折线的点数量
        - coordinates：折线的点坐标列表
        - length：折线的长度

        """
        self.id = index
        self.stroke_id = stroke_id
        self.inter_coords = inter_coords
        self.inter_params = inter_params
        self.adjacent_inter_ids = None


class Intersection3D:
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


# 针对每个stroke，给出所有的3D重建以及相关联信息
class Candidates:
    def __init__(self, candidate_id, stroke_3d, stroke_2d_ids, stroke_2d_inter_ids, stroke_indexes, proxy_info):
        self.candidate_id = candidate_id            # 对应的candidate序号
        self.stroke_3d = stroke_3d                  # 3d重建
        self.stroke_2d_ids = stroke_2d_ids          # symmetric的2d_ids(针对dif_symmetric)
        self.stroke_2d_inter_ids = stroke_2d_inter_ids      # 依据的相交线(仅针对self_symmetric)
        self.stroke_indexes = stroke_indexes        # 重建对应序号
        self.proxy_info = proxy_info                # 相关联的proxy


