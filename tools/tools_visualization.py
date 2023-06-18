import numpy as np

import polyscope as ps

vp_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e6ab02", "#a6761d", "#e7298a"]
vp_colors_rgb = [[27, 158, 119], [217, 95, 2], [117, 112, 179], [231, 41, 138],
                 [102, 166, 30], [230, 171, 2]]


def plot_curve_ps(s_name, s):
    ps.register_curve_network(s_name, np.array(s), np.array([[i, i + 1] for i in range(len(s) - 1)]))


def setup_cam_ps(cam):
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")
    ps.set_up_dir("z_up")
    ps.look_at_dir(camera_location=cam.cam_pos,
                   target=np.array(cam.view_dir) + np.array(cam.cam_pos),
                   up_dir=np.array([0, 0, -1]))


def visualize_polyscope(result, cam):
    ps.init()
    ps.remove_all_structures()

    for s_id, s in enumerate(result):
        if s is not None:
            if len(s[1]) == 0:
                continue
            plot_curve_ps(s[0], s[1])

    setup_cam_ps(cam)
    ps.show()
