import numpy as np
from shapely.geometry import LineString, Point
from skspatial.objects import Plane, Line

from tools import tools_3d as tools_3d


class Camera:
    def __init__(self, proj_mat=None, focal_dist=None, fov=None, view_dir=None,
                 principal_point=None, rot_mat=None, K=None, t=None,
                 cam_pos=None, vanishing_points_coords=None):
        self.proj_mat = proj_mat
        self.focal_dist = focal_dist
        self.fov = fov
        self.view_dir = view_dir
        self.principal_point = principal_point
        self.rot_mat = rot_mat
        self.K = K
        self.t = t
        self.cam_pos = cam_pos
        self.vanishing_points_coords = vanishing_points_coords

    def get_cam_param_dict(self):
        return {
            "P": np.array(self.proj_mat).tolist(),
            "principal_point": np.array(self.principal_point).tolist(),
            "R": np.array(self.rot_mat).tolist(),
            "K": np.array(self.K).tolist(),
            "t": np.array(self.t).tolist(),
            "view_dir": np.array(self.view_dir).tolist(),
            "fov": self.fov,
            "C": np.array(self.cam_pos).tolist(),
            "VP": [1, 2, 3],
            "vp_coord": np.array(self.vanishing_points_coords).tolist()}

    def project_point(self, p):
        hom_p = np.ones(4)
        hom_p[:3] = p
        proj_p = np.dot(self.proj_mat, hom_p)
        return proj_p[:2] / proj_p[2]

    def project_polyline(self, polyline):
        return [self.project_point(p) for p in polyline]

    def get_camera_point_ray(self, p):
        lifted_point = self.lift_point(p, 0.5)
        camera_point_ray = lifted_point - self.cam_pos
        camera_point_ray /= np.linalg.norm(camera_point_ray)
        return lifted_point, camera_point_ray

    # line is a 3D line
    # line_p is a point on the line
    # line_v is the normalized direction vector
    def lift_point_close_to_line(self, p, line_p, line_v, return_axis_point=False):
        lifted_point, camera_point_ray = self.get_camera_point_ray(p)
        closest_point_axis, closest_point_lifted_sketch = \
            tools_3d.line_line_collision(line_p, line_v, lifted_point, camera_point_ray)
        if return_axis_point:
            return closest_point_lifted_sketch, closest_point_axis
        return closest_point_lifted_sketch

    def lift_point_close_to_polyline_v2(self, p, polyline):
        lifted_point, camera_point_ray = self.get_camera_point_ray(p)
        polyline_plane = Plane.best_fit(polyline)
        return polyline_plane.intersect_line(Line(lifted_point, camera_point_ray))

    def lift_point_close_to_polyline(self, p, polyline):
        lifted_point, camera_point_ray = self.get_camera_point_ray(p)
        closest_point_lifted_sketch = \
            tools_3d.line_polyline_collision(np.array(self.cam_pos), camera_point_ray,
                                             polyline)
        return closest_point_lifted_sketch

    def lift_point_close_to_polyline_v3(self, p, polyline, return_axis_point=False):
        lifted_point, camera_point_ray = self.get_camera_point_ray(p)
        proj_polyline = self.project_polyline(polyline)
        seg_dists = [LineString([proj_polyline[p_id], proj_polyline[p_id + 1]]).distance(Point(p)) for p_id in
                     range(len(polyline) - 1)]
        closest_seg = np.array([polyline[np.argmin(seg_dists)], polyline[np.argmin(seg_dists) + 1]])
        closest_point_lifted_sketch, dist = \
            tools_3d.line_segment_collision(np.array(self.cam_pos), camera_point_ray,
                                            closest_seg, return_axis_point=return_axis_point)
        return closest_point_lifted_sketch

    def lift_polyline_close_to_line(self, polyline, line_p, line_v):
        return [self.lift_point_close_to_line(p, line_p, line_v) for p in polyline]

    def lift_point_to_plane(self, p, plane_p, plane_n):
        lifted_point, camera_point_ray = self.get_camera_point_ray(p)
        inter_p = tools_3d.line_plane_collision(plane_normal=plane_n, plane_point=plane_p,
                                                ray_dir=camera_point_ray, ray_p=lifted_point)
        return inter_p

    # polyline_plane = Plane(plane_p, plane_n)
    # return polyline_plane.intersect_line(Line(lifted_point, camera_point_ray))

    def lift_polyline_to_plane(self, polyline, plane_p, plane_n):
        return [self.lift_point_to_plane(p, plane_p, plane_n) for p in polyline]

    def lift_point(self, p, lambda_val):
        u, v = p[0], p[1]

        p_cam = np.dot(self.k_inv, np.array([[u], [v], [1.0]]))
        p_cam *= lambda_val
        p_cam = np.expand_dims(p_cam, 0)

        p_world = np.ones(shape=(4, 1))
        p_world[:3] = p_cam
        p_world = np.dot(self.rot_mat_t_inv, p_world)
        p_world[:] /= p_world[3]
        p_world = p_world[:3]
        p_world = np.transpose(p_world)
        return p_world[0]

    def lift_polyline(self, polyline, lambda_val):
        return [self.lift_point(p, lambda_val) for p in polyline]

    def compute_inverse_matrices(self):

        r_t_inv = np.ones(shape=(4, 4))
        r_t_inv[:3, :3] = self.rot_mat
        r_t_inv[:3, 3] = self.t
        r_t_inv[3, :3] = 0.0
        try:
            r_t_inv = np.linalg.inv(r_t_inv)
        except:
            r_t_inv = np.linalg.pinv(r_t_inv)

        u0 = self.principal_point[0]
        v0 = self.principal_point[1]

        k = np.array([[self.focal_dist, 0.0, u0],
                      [0.0, self.focal_dist, v0],
                      [0.0, 0.0, 1.0]])
        k_inv = np.linalg.inv(k)
        self.rot_mat_t_inv = r_t_inv
        self.k_inv = k_inv
