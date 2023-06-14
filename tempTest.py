# -*- coding:utf-8 -*-
# @Author: IEeya
import numpy as np

from tools.tools_3d import find_closest_points
from ortools.linear_solver import pywraplp


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
    # 创建求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # 定义变量
    x = solver.IntVar(0, 10, 'x')
    y = solver.IntVar(0, 10, 'y')

    # 定义约束条件
    solver.Add(1 <= x+2*y <= 5)

    solver.Add(0 <= 3*x-y <= 10)

    if x <= y:
        final_score = x + y
    else:
        final_score = 2*x + 5*y
    # 定义目标函数
    solver.Maximize(final_score)

    # 求解问题
    status = solver.Solve()

    print(solver.Objective().Value())
    if status == pywraplp.Solver.OPTIMAL:
        print('最优解已找到')
        print('目标函数值 =', final_score)
        print('x 的值 =', x.solution_value())
        print('y 的值 =', y.solution_value())
    else:
        print('求解器未找到最优解。')