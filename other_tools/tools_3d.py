import numpy as np
from skspatial.objects import Plane, Line

ALPHA = np.deg2rad(15.0)
SIGMA_1 = (1.0 - np.cos(ALPHA))/3.0
SIGMA_2 = (np.cos(np.deg2rad(90 - 15)))/3.0


def line_plane_collision(plane_normal, plane_point, ray_dir, ray_p):

    ndotu = plane_normal.dot(ray_dir)
    if np.isclose(abs(ndotu), 0.0):
        raise RuntimeError("no intersection or line is within plane")

    w = ray_p - plane_point
    si = -plane_normal.dot(w) / ndotu
    return w + si * ray_dir + plane_point


def point_line_distance(line_a, line_b, p):
    return np.linalg.norm(np.cross(line_b - line_a, p - line_a))/np.linalg.norm(line_b - line_a)


def line_segment_collision(line_p, line_dir, seg, return_axis_point=False):
    seg_p = seg[0]
    seg_dir = seg[1] - seg[0]
    seg_l = np.linalg.norm(seg_dir)
    if np.isclose(seg_l, 0.0):
        proj_seg_0 = Line(point=line_p, direction=line_dir).project_point(seg[0])
        dist = np.linalg.norm(proj_seg_0-seg[0])
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
        seg_0_dist = point_line_distance(line_p, line_p+1.0*line_dir, seg[0])
        seg_1_dist = point_line_distance(line_p, line_p+1.0*line_dir, seg[1])
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
    return inter_p, point_line_distance(line_p, line_p+1.0*line_dir, inter_p)


def line_polyline_collision(line_p, line_dir, polyline):
    intersections = [line_segment_collision(line_p, line_dir, np.array([polyline[2*i], polyline[2*i+1]]))
                     for i in range(int(len(polyline)/2))]
    #print(intersections)
    intersections = [inter for inter in intersections if inter is not None]
    distances = [inter[1] for inter in intersections]
    if len(distances) == 0:
        return None
    return intersections[np.argmin(distances)][0]


def line_line_collision(p1, v1, p2, v2):
    v3 = np.cross(v1, v2)
    v3 /= np.linalg.norm(v3)

    rhs = p2 - p1
    lhs = np.array([v1, -v2, v3]).T

    t_solutions = np.linalg.lstsq(lhs, rhs, rcond=None)
    #t_solutions = lstsq(lhs, rhs, lapack_driver="gelsy")
    t1 = t_solutions[0][0]
    t2 = t_solutions[0][1]

    closest_line_1 = p1 + t1*v1
    closest_line_2 = p2 + t2*v2
    return [closest_line_1, closest_line_2]


def line_3d_length(polyline):
    l = np.array(polyline)
    if len(l.shape) < 2:
        return 0.0
    try:
        d = np.sum(np.linalg.norm(l[1:] - l[:len(l)-1], axis=1))
    except:
        print(l)
        exit()
    return np.sum(np.linalg.norm(l[1:] - l[:len(l)-1], axis=1))


def get_reflection_mat(plane_point, plane_normal):
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    plane_normal /= np.linalg.norm(plane_normal)
    refl_mat = np.zeros([4, 4])
    for i in range(4):
        refl_mat[i, i] = 1.0
    for i in range(3):
        for j in range(3):
            refl_mat[i, j] -= 2.0*plane_normal[i]*plane_normal[j]
    d = -np.dot(plane_point, plane_normal)
    for i in range(3):
        refl_mat[i, -1] = -2.0*plane_normal[i]*d

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


def plane_plane_intersection(a, b):
    u = np.cross(a[:-1], b[:-1])
    u /= np.linalg.norm(u)
    m = np.array((a[:-1], b[:-1], u))
    x = np.array((-a[-1], -b[-1], 0.))
    return np.linalg.solve(m, x), u


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


def reconstruct_symmetric_strokes_straight(s_1, s_2, sym_plane_point, sym_plane_normal, camera):

    refl_mat = get_reflection_mat(sym_plane_point, sym_plane_normal)
    stroke_triangle = np.array([camera.cam_pos,
                                np.array(camera.lift_point(s_1.lineString.coords[0], 20.0)),
                                np.array(camera.lift_point(s_1.lineString.coords[-1], 20.0))])
    other_stroke_triangle = np.array([camera.cam_pos,
                                      np.array(camera.lift_point(s_2.lineString.coords[0], 20.0)),
                                      np.array(camera.lift_point(s_2.lineString.coords[-1], 20.0))])
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
    line_b = plane_inter_point + 0.1*plane_inter_dir
    reflected_line = apply_hom_transform_to_points(np.array([line_a, line_b]), refl_mat)
    reflected_line_dir = reflected_line[-1] - reflected_line[0]
    reflected_line_dir /= np.linalg.norm(reflected_line_dir)
    inter_line = np.array(camera.lift_polyline_close_to_line([s_2.lineString.coords[0],
                                                              s_2.lineString.coords[-1]],
                                                             line_a, plane_inter_dir))
    final_line = np.array(camera.lift_polyline_close_to_line([s_1.lineString.coords[0],
                                                              s_1.lineString.coords[-1]],
                                                             reflected_line[0], reflected_line_dir))
    return final_line, inter_line


def calculate_intersection(matrix_p, matrix_q):
    # 提取矩阵p中的三个点
    p1, p2, p3 = matrix_p

    # 提取矩阵q中的三个点
    q1, q2, q3 = matrix_q

    # 构建点p1-p2-p3的平面向量
    v1 = p2 - p1
    v2 = p3 - p1

    # 计算点p1-p2-p3的法向量
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)

    # 构建点q1-q2-q3的平面向量
    w1 = q2 - q1
    w2 = q3 - q1

    # 计算点q1-q2-q3的法向量
    normal_vector_q = np.cross(w1, w2)
    normal_vector_q /= np.linalg.norm(normal_vector_q)

    plane1 = Plane(p1, normal_vector)
    plane2 = Plane(p2, normal_vector_q)

    intersection_line = plane1.intersect_plane(plane2)

    return intersection_line
