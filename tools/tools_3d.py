import numpy as np
import open3d as o3d
from math import acos
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy

from skspatial.objects import Line, Plane

ALPHA = np.deg2rad(15.0)
SIGMA_1 = (1.0 - np.cos(ALPHA)) / 3.0
SIGMA_2 = (np.cos(np.deg2rad(90 - 15))) / 3.0


def line_line_intersection_2d(a, b, c, d):
    x = ((a[0] * b[1] - a[1] * b[0]) * (c[0] - d[0]) - (c[0] * d[1] - c[1] * d[0]) * (a[0] - b[0])) \
        / ((a[0] - b[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (a[1] - b[1]))
    y = ((a[0] * b[1] - a[1] * b[0]) * (c[1] - d[1]) - (c[0] * d[1] - c[1] * d[0]) * (a[1] - b[1])) \
        / ((a[0] - b[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (a[1] - b[1]))
    return np.array([x, y])


def line_plane_collision(plane_normal, plane_point, ray_dir, ray_p):
    ndotu = plane_normal.dot(ray_dir)
    if np.isclose(abs(ndotu), 0.0):
        raise RuntimeError("no intersection or line is within plane")

    w = ray_p - plane_point
    si = -plane_normal.dot(w) / ndotu
    return w + si * ray_dir + plane_point


def point_line_distance(line_a, line_b, p):
    return np.linalg.norm(np.cross(line_b - line_a, p - line_a)) / np.linalg.norm(line_b - line_a)


def line_segment_collision(line_p, line_dir, seg, return_axis_point=False):
    seg_p = seg[0]
    seg_dir = seg[1] - seg[0]
    seg_l = np.linalg.norm(seg_dir)
    if np.isclose(seg_l, 0.0):
        proj_seg_0 = Line(point=line_p, direction=line_dir).project_point(seg[0])
        dist = np.linalg.norm(proj_seg_0 - seg[0])
        if return_axis_point:
            return proj_seg_0, dist
        else:
            return seg[0], dist
    seg_dir /= np.linalg.norm(seg_l)
    if return_axis_point:
        inter_p = line_line_collision(line_p, line_dir, seg_p, seg_dir)[0]
    else:
        inter_p = line_line_collision(line_p, line_dir, seg_p, seg_dir)[1]
    a_p = inter_p - seg[0]
    a_p_l = np.linalg.norm(a_p)
    b_p = inter_p - seg[1]
    b_p_l = np.linalg.norm(a_p)
    if np.isclose(a_p_l, 0.0):
        if return_axis_point:
            proj_seg_0 = Line(point=line_p, direction=line_dir).project_point(seg[0])
            return proj_seg_0, a_p_l
        else:
            return seg[0], a_p_l
    if np.isclose(b_p_l, 0.0):
        if return_axis_point:
            proj_seg_1 = Line(point=line_p, direction=line_dir).project_point(seg[1])
            return proj_seg_1, b_p_l
        else:
            return seg[1], b_p_l
    a_p /= a_p_l
    b_p /= b_p_l
    if seg_dir.dot(a_p) <= 0 or seg_dir.dot(b_p) >= 0:
        # intersection is outside of segment
        seg_0_dist = point_line_distance(line_p, line_p + 1.0 * line_dir, seg[0])
        seg_1_dist = point_line_distance(line_p, line_p + 1.0 * line_dir, seg[1])
        if seg_0_dist < seg_1_dist:
            if return_axis_point:
                proj_seg_0 = Line(point=line_p, direction=line_dir).project_point(seg[0])
                return proj_seg_0, seg_0_dist
            else:
                return seg[0], seg_0_dist
        else:
            if return_axis_point:
                proj_seg_1 = Line(point=line_p, direction=line_dir).project_point(seg[1])
                return proj_seg_1, seg_1_dist
            else:
                return seg[1], seg_1_dist
    return inter_p, point_line_distance(line_p, line_p + 1.0 * line_dir, inter_p)


def line_polyline_collision(line_p, line_dir, polyline):
    intersections = [line_segment_collision(line_p, line_dir, np.array([polyline[2 * i], polyline[2 * i + 1]]))
                     for i in range(int(len(polyline) / 2))]
    # print(intersections)
    intersections = [inter for inter in intersections if inter is not None]
    distances = [inter[1] for inter in intersections]
    if len(distances) == 0:
        return None
    return intersections[np.argmin(distances)][0]


# v1 and v2 should be normalized
# returns closest points on line 1 and on line 2
def line_line_collision(p1, v1, p2, v2):
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)

    rhs = p2 - p1
    lhs = np.array([v1, -v2, v3]).T

    t_solutions = np.linalg.lstsq(lhs, rhs, rcond=None)
    # t_solutions = lstsq(lhs, rhs, lapack_driver="gelsy")
    t1 = t_solutions[0][0]
    t2 = t_solutions[0][1]

    closest_line_1 = p1 + t1 * v1
    closest_line_2 = p2 + t2 * v2
    return [closest_line_1, closest_line_2]


# useful to get plane_u, plane_v for an ellipse
def get_basis_for_planar_point_cloud(points):
    points = np.array(points)
    plane = Plane.best_fit(points)
    normal = np.array(plane.normal)
    first_point = points[0]

    max_dist_point = points[np.argmax(np.linalg.norm(points - first_point, axis=-1))]
    plane_u = max_dist_point - first_point
    plane_u /= np.linalg.norm(plane_u)
    plane_v = np.cross(plane_u, normal)
    plane_v /= np.linalg.norm(plane_v)
    return plane_u, plane_v


def get_ellipse_eccentricity(ellipse_3d, plane_u, plane_v):
    projected_ellipse = np.array([[np.dot(plane_u, p), np.dot(plane_v, p)] for p in ellipse_3d])
    x = projected_ellipse[:, 0]
    y = projected_ellipse[:, 1]
    xmean, ymean = x.mean(), y.mean()
    x -= xmean
    y -= ymean
    _, scale, _ = np.linalg.svd(np.stack((x, y)))

    major_axis_length = np.max(scale)
    minor_axis_length = np.min(scale)
    a = major_axis_length / 2
    b = minor_axis_length / 2
    ecc = np.sqrt(np.square(a) - np.square(b)) / a
    return ecc


def angle_line_line(l1, l2):
    vec_1 = l1[-1] - l1[0]
    vec_2 = l2[-1] - l2[0]
    norm_vec_1 = vec_1 / np.linalg.norm(vec_1)
    norm_vec_2 = vec_2 / np.linalg.norm(vec_2)
    if np.isclose(np.dot(norm_vec_1, norm_vec_2), 1.0):
        return 0.0
    return acos(np.abs(np.dot(norm_vec_1, norm_vec_2)))


def line_3d_length(polyline):
    l = np.array(polyline)
    if len(l.shape) < 2:
        return 0.0
    try:
        d = np.sum(np.linalg.norm(l[1:] - l[:len(l) - 1], axis=1))
    except:
        print(l)
        exit()
    return np.sum(np.linalg.norm(l[1:] - l[:len(l) - 1], axis=1))


def merge_two_line_segments(l1, l2):
    # just merge the closest endpoints
    if np.linalg.norm(l1[0] - l2[0]) < np.linalg.norm(l1[0] - l2[1]):
        return np.array([(l1[0] + l2[0]) / 2.0, (l1[1] + l2[1]) / 2.0])
    return np.array([(l1[0] + l2[1]) / 2.0, (l1[1] + l2[0]) / 2.0])


def merge_n_line_segments(lines):
    l1 = lines[0]
    for l in lines[1:]:
        l1 = merge_two_line_segments(l1, l)
    return l1


def get_rotation_mat_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def get_rotation_mat_x(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def get_rotation_mat_y(angle):
    return np.array([[np.cos(angle), 0, -np.sin(angle)],
                     [0, 1, 0],
                     [np.sin(angle), 0, np.cos(angle)]])


def get_reflection_mat(plane_point, plane_normal):
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    plane_normal /= np.linalg.norm(plane_normal)
    refl_mat = np.zeros([4, 4])
    for i in range(4):
        refl_mat[i, i] = 1.0
    for i in range(3):
        for j in range(3):
            refl_mat[i, j] -= 2.0 * plane_normal[i] * plane_normal[j]
    d = -np.dot(plane_point, plane_normal)
    for i in range(3):
        refl_mat[i, -1] = -2.0 * plane_normal[i] * d

    return refl_mat


def apply_hom_transform_to_points(points, hom_mat):
    points = np.array(points)
    hom_points = np.ones([points.shape[0], 4])
    hom_points[:, :-1] = points
    transformed_points = np.dot(hom_mat, hom_points.T)
    transformed_points = transformed_points.transpose()
    transformed_points[:, :] /= transformed_points[:, -1][:, None]
    return transformed_points[:, :-1]


def plane_point_normal_to_equation(plane_point, plane_normal):
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    plane_equation = np.zeros(4)
    plane_equation[:3] = plane_normal[:]
    d = -np.dot(plane_normal, plane_point)
    plane_equation[-1] = d
    return plane_equation


#  a, b: non-parallel plane-equations
def plane_plane_intersection(a, b):
    u = np.cross(a[:-1], b[:-1])
    u /= np.linalg.norm(u)
    m = np.array((a[:-1], b[:-1], u))
    x = np.array((-a[-1], -b[-1], 0.))
    return np.linalg.solve(m, x), u


# def out_of_bbox(bbox, points):
def scale_bbox(bbox, scale_factor):
    bbox_min = bbox[:3]
    bbox_max = bbox[3:]
    bbox_mid = (bbox_min + bbox_max) / 2.0
    bbox_min = bbox_min + (bbox_min - bbox_mid) * scale_factor
    bbox_max = bbox_max + (bbox_max - bbox_mid) * scale_factor
    bbox[:3] = bbox_min
    bbox[3:] = bbox_max


def bbox_from_points(points):
    # min_x, min_y, min_z, max_x, max_y, max_z
    points = np.array(points)
    bbox = np.zeros(6, dtype=float)
    bbox[:3] = 1000.0
    bbox[3:] = -1000.0
    bbox[0] = np.minimum(bbox[0], np.min(points[:, 0]))
    bbox[1] = np.minimum(bbox[1], np.min(points[:, 1]))
    bbox[2] = np.minimum(bbox[2], np.min(points[:, 2]))
    bbox[3] = np.maximum(bbox[3], np.max(points[:, 0]))
    bbox[4] = np.maximum(bbox[4], np.max(points[:, 1]))
    bbox[5] = np.maximum(bbox[5], np.max(points[:, 2]))
    return bbox


def bbox_diag(bbox):
    return np.linalg.norm(bbox[:3] - bbox[3:])


def bbox_volume(bbox):
    return abs(bbox[0] - bbox[3]) * abs(bbox[1] - bbox[4]) * abs(bbox[2] - bbox[5])


def out_of_bbox(points, bbox):
    for p in points:
        if np.any(p < bbox[:3]) or np.any(p > bbox[3:]):
            return True
    return False


def all_out_of_bbox(points, bbox):
    checks = []
    for p in points:
        checks.append(np.any(p < bbox[:3]) or np.any(p > bbox[3:]))
    return np.all(checks)


# seg is an array of points, parametrized by the arc-parameters in interval
# interval: [a, b]
# return point corresponding to arc-parameter t
def interpolate_segment(seg, interval, t):
    if np.isclose(t, interval[0]):
        return seg[0], 0

    if np.isclose(t, interval[1]):
        return seg[-1], len(seg) - 2

    for i in range(len(seg) - 1):
        t_i = interval[0] + \
              i * (interval[1] - interval[0]) / (len(seg) - 1)
        t_i_plus_1 = interval[0] + \
                     (i + 1) * (interval[1] - interval[0]) / (len(seg) - 1)
        if t >= t_i and t <= t_i_plus_1:
            return seg[i] + (t - t_i) / (t_i_plus_1 - t_i) * \
                   (seg[i + 1] - seg[i]), i
    return [], -1


def get_foreshortening(polyline, cam_pos):
    cam_pos = np.array(cam_pos)
    polyline = np.array(polyline)
    cam_dists = [np.linalg.norm(p - cam_pos) for p in polyline]
    foreshortening = np.sum([np.abs(cam_dists[i + 1] - cam_dists[i])
                             for i in range(len(polyline) - 1)])
    return foreshortening


def get_foreshortening_max(polyline, cam_pos):
    cam_pos = np.array(cam_pos)
    polyline = np.array(polyline)
    cam_dists = [np.linalg.norm(p - cam_pos) for p in polyline]
    foreshortening = np.max(cam_dists) - np.min(cam_dists)
    return foreshortening


def reconstruct_two_points(p1, p2, refl_mat, camera):
    p1_lifted = camera.lift_point(p1, 1.0)
    p2_lifted = camera.lift_point(p2, 1.0)
    p1_projective = np.array([camera.cam_pos, p1_lifted])
    p2_projective = np.array([camera.cam_pos, p2_lifted])
    p2_projective /= np.linalg.norm(p2_projective)
    reflected_p1_projective = apply_hom_transform_to_points(p1_projective, refl_mat)
    reflected_p1_projective_dir_vec = reflected_p1_projective[-1] - reflected_p1_projective[0]
    reflected_p1_projective_dir_vec /= np.linalg.norm(reflected_p1_projective_dir_vec)
    p2_projective_dir_vec = p2_projective[-1] - p2_projective[0]
    p2_projective_dir_vec /= np.linalg.norm(p2_projective_dir_vec)
    p2_reconstructed, _ = line_line_collision(reflected_p1_projective[-1], reflected_p1_projective_dir_vec,
                                              p2_lifted, p2_projective_dir_vec)
    p2_reconstructed = np.array(p2_reconstructed)
    p1_reconstructed = apply_hom_transform_to_points([p2_reconstructed], refl_mat)[0]
    return [p1_reconstructed, p2_reconstructed]


# returns 2 arrays of shape (2, 3), i.e., two 3d lines
def reconstruct_symmetric_strokes_straight(s_1, s_2, sym_plane_point, sym_plane_normal, camera):
    refl_mat = get_reflection_mat(sym_plane_point, sym_plane_normal)
    stroke_triangle = np.array([camera.cam_pos,
                                np.array(camera.lift_point(s_1.points_list[0].coords, 20.0)),
                                np.array(camera.lift_point(s_1.points_list[-1].coords, 20.0))])
    other_stroke_triangle = np.array([camera.cam_pos,
                                      np.array(camera.lift_point(s_2.points_list[0].coords, 20.0)),
                                      np.array(camera.lift_point(s_2.points_list[-1].coords, 20.0))])
    reflected_stroke_triangle = apply_hom_transform_to_points(stroke_triangle, refl_mat)
    # intersection between both triangles
    first_triangle_point = reflected_stroke_triangle[0]
    first_triangle_normal = np.cross(reflected_stroke_triangle[1] - reflected_stroke_triangle[0],
                                     reflected_stroke_triangle[2] - reflected_stroke_triangle[0])
    first_triangle_normal /= np.linalg.norm(first_triangle_normal)
    first_triangle_plane_equation = plane_point_normal_to_equation(
        first_triangle_point, first_triangle_normal)
    snd_triangle_point = stroke_triangle[0]
    snd_triangle_normal = np.cross(other_stroke_triangle[1] - other_stroke_triangle[0],
                                   other_stroke_triangle[2] - other_stroke_triangle[0])
    snd_triangle_normal /= np.linalg.norm(snd_triangle_normal)
    snd_triangle_plane_equation = plane_point_normal_to_equation(
        snd_triangle_point, snd_triangle_normal)

    plane_inter_point, plane_inter_dir = plane_plane_intersection(
        first_triangle_plane_equation, snd_triangle_plane_equation)
    inter_line = Plane(first_triangle_point, first_triangle_normal).intersect_plane(
        Plane(snd_triangle_point, snd_triangle_normal))
    plane_inter_point = inter_line.point
    plane_inter_dir = inter_line.vector
    line_a = plane_inter_point
    line_b = plane_inter_point + 0.1 * plane_inter_dir
    reflected_line = apply_hom_transform_to_points(np.array([line_a, line_b]), refl_mat)
    reflected_line_dir = reflected_line[-1] - reflected_line[0]
    reflected_line_dir /= np.linalg.norm(reflected_line_dir)
    inter_line = np.array(camera.lift_polyline_close_to_line([s_2.points_list[0].coords,
                                                              s_2.points_list[-1].coords],
                                                             line_a, plane_inter_dir))
    final_line = np.array(camera.lift_polyline_close_to_line([s_1.points_list[0].coords,
                                                              s_1.points_list[-1].coords],
                                                             reflected_line[0], reflected_line_dir))
    return final_line, inter_line


def chamfer_distance(x, y, metric='l2', direction='bi', return_pointwise_distances=False):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    pointwise_dists = []
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
        pointwise_dists = min_y_to_x
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
        pointwise_dists = min_x_to_y
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        pointwise_dists = [min_x_to_y, min_y_to_x]
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    if return_pointwise_distances:
        return chamfer_dist, pointwise_dists
    return chamfer_dist


def icp_registration(s1_pts, s2_pts, with_scaling=False):
    initial_T = np.identity(4)  # Initial transformation for ICP
    source = np.zeros([len(s1_pts), 3])
    source[:, :2] = s1_pts
    target = np.zeros([len(s2_pts), 3])
    target[:, :2] = s2_pts
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target)

    distance = 1000.0  # The threshold distance used for searching correspondences
    icp_type = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling)
    iterations = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
    result = o3d.pipelines.registration.registration_icp(source_pc, target_pc, distance, initial_T, icp_type,
                                                         iterations)
    trans = np.asarray(deepcopy(result.transformation))
    # if trans[0, 0]*trans[1, 1] < 0:
    #    trans[:, 0] *= -1
    source_pc.transform(trans)
    scale_x = np.linalg.norm(trans[:, 0])
    scale_y = np.linalg.norm(trans[:, 1])
    scale_z = np.linalg.norm(trans[:, 2])
    scale_z = trans[2, 2]
    reflection = scale_z < 0
    rot_mat = deepcopy(trans)
    rot_mat[:, 0] /= scale_z
    rot_mat[:, 1] /= scale_z
    rot_mat[:, 2] /= scale_z
    angle = np.rad2deg(np.arccos(rot_mat[0, 0]))

    return np.array(source_pc.points)[:, :2], angle, reflection


def rotate_angle(pts, angle):
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return np.matmul(rot_mat, pts.T).T
