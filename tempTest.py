# -*- coding:utf-8 -*-
# @Author: IEeya
import numpy as np

from tools.tools_3d import find_closest_points


def apply_hom_transform_to_points(points, hom_mat):
    points = np.array(points)
    hom_points = np.ones([points.shape[0], 4])
    hom_points[:, :-1] = points
    transformed_points = np.dot(hom_mat, hom_points.T)
    transformed_points = transformed_points.transpose()
    transformed_points[:, :] /= transformed_points[:, -1][:, None]
    return transformed_points[:, :-1]


# 这个函数的作用是获得一个点相对于相机点(plane_point的两个参数)关于面plane_normal的对称结果
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


if __name__ == '__main__':
    sym_plane_point = np.zeros(3, dtype=np.float_)
    sym_plane_normal = np.zeros(3, dtype=np.float_)
    sym_plane_normal[0] = 1.0
    refl_mat = get_reflection_mat(sym_plane_point, sym_plane_normal)

    p1_lifted = (5.0, 0.0, 0.0)
    p2_lifted = (3.0, 4.0, 0.0)
    cam_pos = (1.0, 1.0, 2.0)

    p1_projective = np.array([cam_pos, p1_lifted])
    reflected_p1_projective = apply_hom_transform_to_points(p1_projective, refl_mat)
    reflected_p1_projective_dir_vec = reflected_p1_projective[-1] - reflected_p1_projective[0]
    reflected_p1_projective_dir_vec /= np.linalg.norm(reflected_p1_projective_dir_vec)

    p2_projective = np.array([cam_pos, p2_lifted])
    p2_projective /= np.linalg.norm(p2_projective)
    p2_projective_dir_vec = p2_projective[-1] - p2_projective[0]
    p2_projective_dir_vec /= np.linalg.norm(p2_projective_dir_vec)

    # 在这里，存储了P1和P2的向量信息（出发点，以及方向向量，下面的函数用于构建P1和P2之间距离最近的两个点的位置）
    p2_reconstructed, _ = line_line_collision(reflected_p1_projective[-1], reflected_p1_projective_dir_vec,
                                              p2_lifted, p2_projective_dir_vec)
    p3_reconstructed, _ = find_closest_points(reflected_p1_projective[-1], reflected_p1_projective_dir_vec,
                                              p2_lifted, p2_projective_dir_vec)
    p2_reconstructed = np.array(p2_reconstructed)
    p1_reconstructed = apply_hom_transform_to_points([p2_reconstructed], refl_mat)[0]