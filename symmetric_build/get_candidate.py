# -*- coding:utf-8 -*-
# @Author: IEeya
# 该函数主要用于从Sketch中获取全部的候选组
import numpy as np
from scipy.spatial import ConvexHull
from shapely import Polygon, LineString, MultiLineString, MultiPoint

from tools import tools_3d
from tools.tools_3d import apply_hom_transform_to_points, get_reflection_mat, line_line_collision, line_3d_length, \
    calculate_intersection

# 分组
param_middle = 0.5
deviation = 0.001
# 间隔阈值，大于3/10的线段长度
threshold = 0.3

candidates_3d = []


def get_all_candidates_from_sketch(sketch, cam):
    # 主视角
    main_dir = [0, 1, 2]
    max_dir = [0, 1, 2, 3]
    # 遍历所有视角，第一轮，遍历主视角方向的线
    for vp_dir in main_dir:
        # 遍历所有线
        # 观测方向和遍历线的轴朝向一致的情况
        get_candidates_from_self_v2(cam, vp_dir, sketch)
    # 观测方向和遍历线的轴朝向不一致的情况
    for vp_dir in main_dir:
        for axis_dir in max_dir:
            if vp_dir != axis_dir:
                get_candidates_from_dif_dir_v2(cam, vp_dir, axis_dir, sketch)
    return candidates_3d


# 获取所有的合理的交叉点组合以及其中点投影坐标
def get_candidates_from_same_dir(cam, stroke, sketch):
    # 获取
    main_axis = stroke.axis_label

    intersect_infos = sketch.intersect_dict[stroke.id]
    intersect_params_info = []
    # 得到所有的intersect_params情况
    for intersect_info_id in intersect_infos:
        intersect_info = sketch.intersect_infor[intersect_info_id]
        intersect_stroke_id = intersect_info.stroke_id[1]
        intersect_params_info.append(
            [intersect_stroke_id, intersect_info.inter_coords, intersect_info.inter_params[0]])

    left_side = []
    right_side = []
    for param in intersect_params_info:
        if param[2] < param_middle - deviation:
            left_side.append(param)
        elif param[2] > param_middle + deviation:
            right_side.append(param)
    # 判断，形成candidate
    # item: [id, mid_param]
    candidates_2d = [[left_item, right_item]
                     for left_item in left_side
                     for right_item in right_side
                     if sketch.strokes[left_item[0]].axis_label == sketch.strokes[right_item[0]].axis_label and
                     abs(left_item[2] - right_item[2]) > threshold
                     ]
    """
        candidates_2d :[left, right]
        left: [id, inter_coords, mid_param]
        right: [id, inter_coords, mid_param]
    """
    plane = np.zeros(3, dtype=np.float_)
    plane[main_axis] = 1.0
    refl_mat = get_reflection_mat(np.zeros(3, dtype=np.float_), plane)
    # 3D 化
    for item in candidates_2d:
        p1_lifted = cam.lift_point(item[0][1], 1.0)
        p2_lifted = cam.lift_point(item[1][1], 1.0)
        p1_projective = np.array([cam.cam_pos, p1_lifted])
        p2_projective = np.array([cam.cam_pos, p2_lifted])
        # 归一化
        p2_projective_dir = p2_projective[1] - p2_projective[0]
        p2_projective_dir /= np.linalg.norm(p2_projective_dir)
        # 平面对称
        reflected_p1_projective = apply_hom_transform_to_points(p1_projective, refl_mat)
        # 获取方向
        reflected_p1_projective_dir = reflected_p1_projective[1] - reflected_p1_projective[0]
        # 归一化
        reflected_p1_projective_dir /= np.linalg.norm(reflected_p1_projective_dir)
        # 找到rC rp1直线与C, p2直线的最佳交点
        answer = line_line_collision(reflected_p1_projective[1], reflected_p1_projective_dir,
                                     p2_projective[1], p2_projective_dir)
        # 针对重建的p2点，对称回来，得到重建的p1点
        res_p2 = answer[0]
        res_p1 = apply_hom_transform_to_points([res_p2], refl_mat)[0]
        # 得到重建直线方向
        res_dir = res_p2 - res_p1
        res_dir /= np.linalg.norm(res_dir)
        # 与原直线求最近端点
        answer = np.array(cam.lift_polyline_close_to_line(
            [np.array(stroke.lineString.coords[0]), np.array(stroke.lineString.coords[-1])],
            res_p1, res_dir))
        candidates_3d.append([
            stroke.id, stroke.id,
            answer, answer,
            main_axis,
            -1, -1,
            item[0][0], item[1][0]
        ])


def get_candidates_from_dif_dir(cam, vp_dir, axis_dir, sketch):
    # 筛选出消失点方向的线
    select_stroke = []
    plane = np.zeros(3, dtype=np.float_)
    plane[vp_dir] = 1.0
    refl_mat = get_reflection_mat(np.zeros(3, dtype=np.float_), plane)

    for stroke in sketch.strokes:
        # 非消失点方向的线，排除
        if stroke.axis_label != axis_dir:
            continue
        # 构建消失三角形
        # 获取消失点
        points_list = list(np.array(stroke.lineString.coords))
        vanishing_point = cam.vanishing_points_coords[vp_dir]
        points_list.append(vanishing_point)
        hull = ConvexHull(points_list)
        # 获取凸包的顶点索引
        hull_vertices = hull.vertices
        # 获取凸包的顶点坐标
        hull_index = [points_list[vertex] for vertex in hull_vertices]
        vanishing_polygon = Polygon(hull_index).simplify(5)
        # 生成候选序列
        candidate_stroke = [s for s in sketch.strokes if s.id < stroke.id and s.axis_label == stroke.axis_label]
        for item in candidate_stroke:
            # 先判断距离
            if stroke.calculate_dis_different(item) <= threshold * stroke.calculate_length():
                continue
            # 判断与消失三角形的相交情况
            if not vanishing_polygon.intersects(item.lineString):
                continue
            # 判断相交图形是否为LineString或者MultiLineString
            intersect = vanishing_polygon.intersection(item.lineString)
            if not (type(intersect) == LineString or type(intersect) == MultiLineString):
                continue
            select_stroke.append([stroke, item])

    # 三维化
    for item in select_stroke:
        p1_triangle = np.array([cam.cam_pos,
                                cam.lift_point(item[0].lineString.coords[0], 20.0),
                                cam.lift_point(item[0].lineString.coords[-1], 20.0)
                                ])
        p2_triangle = np.array([cam.cam_pos,
                                cam.lift_point(item[1].lineString.coords[0], 20.0),
                                cam.lift_point(item[1].lineString.coords[-1], 20.0)
                                ])
        # 对称三角形
        reflected_p1_triangle = apply_hom_transform_to_points(p1_triangle, refl_mat)
        # 获取交平面的直线
        inter_line = calculate_intersection(reflected_p1_triangle, p2_triangle)
        # 使用点和向量构建直线
        reflected_line = apply_hom_transform_to_points([inter_line.point, inter_line.point + inter_line.vector],
                                                       refl_mat)
        reflected_line_dir = reflected_line[-1] - reflected_line[0]
        reflected_line_dir /= np.linalg.norm(reflected_line_dir)

        # 与相机重构
        inter_line_2 = cam.lift_polyline_close_to_line(
            [np.array(item[1].lineString.coords[0]), np.array(item[1].lineString.coords[-1])],
            inter_line.point, inter_line.vector)
        inter_line_1 = cam.lift_polyline_close_to_line(
            [np.array(item[0].lineString.coords[0]), np.array(item[0].lineString.coords[-1])],
            reflected_line[0], reflected_line_dir)

        inter_line_1_dir = inter_line_1[-1] - inter_line_1[0]
        inter_line_1_dir /= np.linalg.norm(inter_line_1_dir)
        inter_line_2_dir = inter_line_2[-1] - inter_line_2[0]
        inter_line_2_dir /= np.linalg.norm(inter_line_2_dir)

        final_line_1 = cam.lift_polyline_close_to_line(
            [np.array(item[0].lineString.coords[0]), np.array(item[0].lineString.coords[-1])],
            inter_line_1[0], inter_line_1_dir)
        final_line_2 = cam.lift_polyline_close_to_line(
            [np.array(item[1].lineString.coords[0]), np.array(item[1].lineString.coords[-1])],
            inter_line_2[0], inter_line_2_dir)

        # 过滤小于0.33或者0.66比例的线
        length_ratio = line_3d_length(final_line_1) / line_3d_length(final_line_2)
        if 0.33 < length_ratio < 1.66:
            candidates_3d.append([
                item[0].id, item[1].id,
                final_line_1, final_line_2,
                vp_dir,
                -1, -1,
                -1, -1,
            ])


def get_candidates_from_self_v2(camera, vp_dir, sketch):
    focus_vp = vp_dir
    plane = np.zeros(3, dtype=np.float_)
    plane[focus_vp] = 1.0
    refl_mat = get_reflection_mat(np.zeros(3, dtype=np.float_), plane)
    for s_id, s in enumerate(sketch.strokes):
        # no self-symmetric curves!
        if s.axis_label != focus_vp:
            continue
        candidate_tuples, candidate_tuple_s_ids = get_self_symmetry_candidates(
            s,
            [sketch.intersect_infor[inter_id] for inter_id in sketch.intersect_dict[s.id]],
            sketch
        )
        if len(candidate_tuples) == 0:
            continue
        # 3D candidates
        candidate_reconstructions = np.array(
            [tools_3d.reconstruct_two_points(p1, p2, refl_mat, camera)
             for p1, p2 in candidate_tuples])
        for rec_id, rec in enumerate(candidate_reconstructions):
            rec_vec = rec[-1] - rec[0]
            rec_vec /= np.linalg.norm(rec_vec)
            rec = np.array(camera.lift_polyline_close_to_line([s.lineString.coords[0],
                                                               s.lineString.coords[-1]],
                                                              rec[0], rec_vec))
            # proxy-ids will get filled in later
            candidates_3d.append([s_id, s_id, rec, rec, focus_vp, -1, -1,
                                  candidate_tuple_s_ids[rec_id][0],
                                  candidate_tuple_s_ids[rec_id][1]])


def get_candidates_from_dif_dir_v2(camera, vp_dir, axis_dir, sketch):
    focus_axis_label = axis_dir
    focus_vp = vp_dir
    sym_plane_point = np.zeros(3, dtype=np.float_)
    sym_plane_normal = np.zeros(3, dtype=np.float_)
    sym_plane_normal[focus_vp] = 1.0

    tmp_candidate_correspondences = gather_symmetry_correspondences_2d(sketch, camera,
                                                                       overlap_thresh=0.4,
                                                                       focus_axis_label=focus_axis_label,
                                                                       focus_vp=focus_vp)

    tmp_candidate_correspondences = np.array(tmp_candidate_correspondences)

    for corr_id, corr in enumerate(tmp_candidate_correspondences):
        add_corr = True
        # we cannot compute curve geometry right now
        s_1, s_2 = tools_3d.reconstruct_symmetric_strokes_straight(
            sketch.strokes[corr[0]], sketch.strokes[corr[1]],
            sym_plane_point, sym_plane_normal, camera)
        # reproject input stroke on candidate
        try:
            s_1_dir_vec = s_1[-1] - s_1[0]
            s_1_dir_vec /= np.linalg.norm(s_1_dir_vec)
            s_1 = np.array(camera.lift_polyline_close_to_line([sketch.strokes[corr[0]].lineString.coords[0],
                                                               sketch.strokes[corr[0]].lineString.coords[-1]],
                                                              s_1[0], s_1_dir_vec))
            s_2_dir_vec = s_2[-1] - s_2[0]
            s_2_dir_vec /= np.linalg.norm(s_2_dir_vec)
            s_2 = np.array(camera.lift_polyline_close_to_line([sketch.strokes[corr[1]].lineString.coords[0],
                                                               sketch.strokes[corr[1]].lineString.coords[-1]],
                                                              s_2[0], s_2_dir_vec))
            length_ratio = tools_3d.line_3d_length(s_1) / tools_3d.line_3d_length(s_2)
            if length_ratio < 0.33 or length_ratio > 1.66:
                add_corr = False
        except:
            add_corr = False
            # continue
        # proxy-ids will get filled in later
        if add_corr:
            candidates_3d.append([corr[0], corr[1], s_1, s_2, focus_vp, -1, -1, -1, -1])


def gather_symmetry_correspondences_2d(sketch, camera, overlap_thresh=0.7,
                                       focus_vp=0, focus_axis_label=1):
    intersected_stroke_ids = []
    straight_line_overlap_thresh = overlap_thresh
    for s_id, s in enumerate(sketch.strokes):
        if s.axis_label == 5:
            continue
        else:
            overlap_thresh = straight_line_overlap_thresh
        if s.axis_label != focus_axis_label:
            continue

        s_coords = [p for p in s.lineString.coords]

        vanishing_triangles = \
            [MultiPoint(s_coords + [vp_i]).convex_hull
             for i, vp_i in enumerate(camera.vanishing_points_coords) if not i == s.axis_label and i == focus_vp]
        # extract vanishing_triangle for each convex poly
        for vanishing_triangle_id, vanishing_triangle in enumerate(vanishing_triangles):
            vp_id = -1
            for p_id, p in enumerate(np.array(vanishing_triangle.exterior.coords)):
                found_vp = False
                for vp in camera.vanishing_points_coords:
                    if np.isclose(np.linalg.norm(p - vp), 0.0):
                        vp_id = p_id
                        found_vp = True
                        break
                if found_vp:
                    break
            triangle = np.array(vanishing_triangle.exterior.coords)[:-1][
                [(vp_id - 1) % len(np.array(vanishing_triangle.exterior.coords)[:-1]), vp_id,
                 (vp_id + 1) % len(np.array(vanishing_triangle.exterior.coords)[:-1])]]
            triangle[0] += (triangle[0] - triangle[1])
            triangle[-1] += (triangle[-1] - triangle[1])
            vanishing_triangles[vanishing_triangle_id] = Polygon(triangle)

        for other_s_id in range(len(sketch.strokes)):
            if other_s_id >= s_id:
                continue
            other_s = sketch.strokes[other_s_id]
            if other_s.axis_label != s.axis_label:
                continue

            if s.axis_label < 3 and \
                    other_s.lineString.distance(s.lineString) < 5:
                continue
            if s.axis_label > 3 and \
                    (s.lineString.buffer(5).intersection(
                        other_s.lineString).length / s.length > 0.5 or
                     s.lineString.buffer(5).intersection(
                         other_s.lineString).length / other_s.length > 0.5):
                continue

            for i, vanishing_triangle in enumerate(vanishing_triangles):

                if not vanishing_triangle.intersects(sketch.strokes[other_s_id].lineString):
                    continue
                other_intersecting_segment = vanishing_triangle.intersection(sketch.strokes[other_s_id].lineString)
                if not (type(other_intersecting_segment) == LineString or type(
                        other_intersecting_segment) == MultiLineString):
                    continue
                if other_intersecting_segment.length / sketch.strokes[other_s_id].length > overlap_thresh:
                    intersected_stroke_ids.append([s_id, other_s_id])
    return intersected_stroke_ids


# s is a pystroke.Stroke
# intersections is a list of pystroke.EdgeIntersection
def get_self_symmetry_candidates(s, intersections, sketch, VERBOSE=False):
    # only do combinations of intersections between intersections at endpoints
    # of stroke
    inter_params = [np.array(inter.inter_params)[np.array(inter.stroke_id) == s.id][0]
                    for inter in intersections]
    inter_other_stroke_ids = [np.array(inter.stroke_id)[np.array(inter.stroke_id) != s.id][0]
                              for inter in intersections]
    inter_params_middle = 0.5
    start_points = [inter.inter_coords
                    for inter_id, inter in enumerate(intersections)
                    if inter_params[inter_id] < inter_params_middle - 0.001]
    start_points_other_stroke_ids = [inter_other_stroke_ids[inter_id]
                                     for inter_id, inter in enumerate(intersections)
                                     if inter_params[inter_id] < inter_params_middle - 0.001]
    end_points = [inter.inter_coords
                  for inter_id, inter in enumerate(intersections)
                  if inter_params[inter_id] > inter_params_middle + 0.001]
    end_points_other_stroke_ids = [inter_other_stroke_ids[inter_id]
                                   for inter_id, inter in enumerate(intersections)
                                   if inter_params[inter_id] > inter_params_middle + 0.001]
    candidate_tuples = [[p_start, p_end]
                        for p_start_id, p_start in enumerate(start_points)
                        for p_end_id, p_end in enumerate(end_points)
                        if (start_points_other_stroke_ids[p_start_id] == -1 or \
                            end_points_other_stroke_ids[p_end_id] == -1 or \
                            sketch.strokes[start_points_other_stroke_ids[p_start_id]].axis_label == sketch.strokes[
                                end_points_other_stroke_ids[p_end_id]].axis_label)
                        and np.linalg.norm(np.array(p_start) - np.array(p_end)) > 5]
    candidate_tuple_s_ids = [[start_points_other_stroke_ids[p_start_id], end_points_other_stroke_ids[p_end_id]]
                             for p_start_id, p_start in enumerate(start_points)
                             for p_end_id, p_end in enumerate(end_points)
                             if (start_points_other_stroke_ids[p_start_id] == -1 or \
                                 end_points_other_stroke_ids[p_end_id] == -1 or \
                                 sketch.strokes[start_points_other_stroke_ids[p_start_id]].axis_label == sketch.strokes[
                                     end_points_other_stroke_ids[p_end_id]].axis_label)
                             and np.linalg.norm(np.array(p_start) - np.array(p_end)) > 5]

    return candidate_tuples, candidate_tuple_s_ids
