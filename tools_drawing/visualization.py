import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiLineString, LineString, MultiPoint
from shapely.plotting import plot_polygon

from matplotlib.patches import Circle

import polyscope as ps
import os
import networkx as nx
import seaborn as sns

from symmetric_build.common_tools import extract_fixed_strokes
from tools import tools_3d
from tools_drawing.verbose_decorator import verbose

vp_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e6ab02", "#a6761d", "#e7298a"]
vp_colors_rgb = [[27,158,119], [217,95,2], [117,112,179], [231,41,138],
    [102,166,30], [230,171,2]]
my_dpi = 96.0
patch_blue = "#B2CBE5"

def get_sketch_limits(sketch):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
                        bottom=0.0,
                        top=1.0)
    sketch.display_strokes_2(fig=fig, ax=axes, color_process=lambda s: "black")
    axes.set_xlim(0, sketch.width)
    axes.set_ylim(sketch.height, 0)
    axes.axis("equal")
    axes.axis("off")
    x_lim = axes.get_xlim()
    y_lim = axes.get_ylim()
    plt.close(fig)
    return x_lim, y_lim

def plot_end(sketch, fig, axes, file_name, with_sketch_limits=False,
    VERBOSE=False):
    if with_sketch_limits:
        x_lim, y_lim = get_sketch_limits(sketch)
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)
        axes.set_xlim(0, sketch.width)
        axes.set_ylim(sketch.height, 0)
    else:
        axes.invert_yaxis()
        axes.axis("equal")
        axes.axis("off")
    if not VERBOSE:
        print(1)
    else:
        plt.show()

def plot_curve_ps(s_name, s):
    ps.register_curve_network(s_name,
        np.array(s), np.array([[i, i+1] for i in range(len(s)-1)]))

def setup_cam_ps(cam):
    ps.set_ground_plane_mode("shadow_only")
    ps.set_navigation_style("free")
    #ps.set_up_dir("neg_z_up")
    ps.set_up_dir("z_up")
    ps.look_at_dir(camera_location=cam.cam_pos,
                   target=np.array(cam.view_dir) + np.array(cam.cam_pos), 
                   up_dir=np.array([0, 0, -1]))

def get_sketch_fig(sketch):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
                        bottom=0.0,
                        top=1.0)
    return fig, axes

def sketch_plot(sketch, VERBOSE=False):
    fig, axes = get_sketch_fig(sketch)
    sketch.display_strokes_2(fig=fig, ax=axes, norm_global=True,
        color_process=lambda s: "black", linewidth_data=lambda s:3)
        #color_process = lambda s:s.get_data("pressure"))
        #linewidth_process=lambda p: p.get_data("pressure"))
    plot_end(sketch, fig, axes, "input", VERBOSE=VERBOSE, with_sketch_limits=True)

def clean_folder(folder):
    if os.path.exists(folder):
        test = os.listdir(folder)
        for item in test:
            if item.endswith(".png"):
                os.remove(os.path.join(folder, item))
    else:
        os.mkdir(folder)

def sketch_plot_successive(sketch, folder):

    clean_folder(folder)
    for s_id, s in enumerate(sketch.strokes):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
                        bottom=0.0,
                        top=1.0)

        pts = np.array([p.coords for p in s.points_list])
        axes.plot(pts[:, 0], pts[:, 1], lw=3.0, c="black")
        for prev_s_id, prev_s in enumerate(sketch.strokes[:s_id]):
            pts = np.array([p.coords for p in prev_s.points_list])
            axes.plot(pts[:, 0], pts[:, 1], lw=1.0, c="grey")

        #axes.set_xlim(0, max(sketch.width, sketch.height))
        #axes.set_ylim(max(sketch.width, sketch.height), 0)
        axes.axis("off")
        axes.axis("equal")
        axes.invert_yaxis()
        tmp_file_name = os.path.join(folder, str(np.char.zfill(str(s_id), 3))+".png")
        fig.set_size_inches((512 / my_dpi, 512 / my_dpi))
        plt.savefig(tmp_file_name)
        plt.close(fig)

def get_vanishing_points_fig(sketch, cam, VERBOSE=False):
    # plot lines
    fig, axes = get_sketch_fig(sketch)
    # plot vanishing points
    axes.scatter(np.array(cam.vanishing_points_coords)[:, 0],
                 np.array(cam.vanishing_points_coords)[:, 1], c=vp_colors[:3])
    # plot extended lines
    vp_triangle = MultiLineString([
        [cam.vanishing_points_coords[0], cam.vanishing_points_coords[1]],
        [cam.vanishing_points_coords[1], cam.vanishing_points_coords[2]],
        [cam.vanishing_points_coords[2], cam.vanishing_points_coords[0]]
        ])
    for i, s in enumerate(sketch.strokes):
        if s.is_curved():
            continue
        p0 = np.array(s.points_list[0].coords)
        v = p0 - np.array(s.points_list[-1].coords)
        ext_line = LineString([p0 - 1000*v, p0 + 1000*v])
        intersections = ext_line.intersection(vp_triangle)
        plot_ext_line = []
        for inter in intersections.geoms:
            plot_ext_line.append(np.array(inter.coords[0]))
        plot_ext_line = np.array(plot_ext_line)
        if len(plot_ext_line.shape) == 2:
            axes.plot(plot_ext_line[:,0], plot_ext_line[:,1], c=vp_colors[s.axis_label],
            linestyle="dashed")
    sketch.display_strokes_2(fig=fig, ax=axes, norm_global=True,
        color_process=lambda s:vp_colors[s.axis_label], linewidth_data=lambda s: 3)
    return fig, axes

@verbose
def vanishing_points(sketch, cam, VERBOSE=False):
    fig, axes = get_vanishing_points_fig(sketch, cam, VERBOSE)
    plot_end(sketch, fig, axes, "vanishing_points.png", VERBOSE)
    fig, axes = get_vanishing_points_fig(sketch, cam, VERBOSE)
    plot_end(sketch, fig, axes, "vanishing_points_zoom.png", with_sketch_limits=True, 
        VERBOSE=VERBOSE)

def symm_candidates_ps(sketch, correspondences, cam):
    points = []
    ps.init()
    ps.remove_all_structures()
    for corr_id, corr in enumerate(correspondences):
        s1_id = corr[0]
        s2_id = corr[1]
        s1 = np.array(corr[2])
        s2 = np.array(corr[3])
        axis_id = np.array(corr[4])
        plot_curve_ps(str(s1_id), s1)
        plot_curve_ps(str(s2_id), s2)
    setup_cam_ps(cam)
    ps.show()


def visualize_polyscope(batches, cam):
    if len(batches) == 0:
        return
    fixed_strokes = extract_fixed_strokes(batches)
    ps.init()
    ps.remove_all_structures()
    for s_id, s in enumerate(fixed_strokes):
        if len(s) == 0:
            continue
        plot_curve_ps(str(s_id), s)

    batches_plane_points, plane_faces = get_planes_3d(batches)
    for batch_id, plane_points in enumerate(batches_plane_points):
        for i in range(3):
            ps.register_surface_mesh("batch_"+str(batch_id)+"_"+str(i)+"_plane", 
                np.array(plane_points[i]), np.array(plane_faces), 
                transparency=0.4, color=np.array(vp_colors_rgb[i])/255)
    
    setup_cam_ps(cam)
    ps.show()

@verbose
def plot_acc_radius(sketch, VERBOSE=False):
    fig, axes = get_sketch_fig(sketch)
    for s in sketch.strokes:
        if np.isclose(s.acc_radius, 0.0):
            continue
        #patch = PolygonPatch(s.linestring.linestring.buffer(s.acc_radius).coords, fc=patch_blue, ec=patch_blue, alpha=0.5, zorder=0)
        #axes.add_patch(patch)
        plot_polygon(s.linestring.linestring.buffer(s.acc_radius), ax=axes, 
            facecolor=patch_blue, edgecolor=patch_blue, alpha=0.5, zorder=0, add_points=False)
    sketch.display_strokes_2(fig=fig, ax=axes, norm_global=True,
        color_process=lambda s:"black", linewidth_data=lambda s: 3)
    plot_end(sketch, fig, axes, "acc_radius", VERBOSE)

@verbose
def plot_stroke_type(sketch, VERBOSE=False):
    fig, axes = get_sketch_fig(sketch)
    stroke_types = []
    for s_id, s in enumerate(sketch.strokes):
        if np.isclose(s.length(), 0.0):
            stroke_types.append(3)
        elif not s.is_curved():
            stroke_types.append(0)
        elif s.is_ellipse():
            stroke_types.append(2)
        else:
            stroke_types.append(1)
    sketch.display_strokes_2(fig=fig, ax=axes, norm_global=True,
        color_process=lambda s:vp_colors[stroke_types[s.id]], linewidth_data=lambda s: 3)
    plot_end(sketch, fig, axes, "stroke_types", with_sketch_limits=True)


def visualize_batches(sketch, batches):
    batches_folder = os.path.join(sketch.sketch_folder, "batches")
    clean_folder(batches_folder)
    for batch_id, batch in enumerate(batches):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
                            bottom=0.0,
                            top=1.0)
        sketch.display_strokes_2(fig=fig, ax=axes,
                                 color_process=lambda s: "grey")
        sketch.display_strokes_2(fig=fig, ax=axes,
                                 display_strokes=np.arange(batch[0], batch[1]+1),
                                 color_process=lambda s: "black",
                                 linewidth_data=lambda s: 3.0)
        #axes.set_xlim(0, sketch.width)
        #axes.set_ylim(sketch.height, 0)
        axes.axis("equal")
        axes.axis("off")
        axes.invert_yaxis()
        fig.set_size_inches((1024 / my_dpi, 1024 / my_dpi))
        plt.savefig(os.path.join(batches_folder, "batch_"+str(batch_id)+".png"))
        plt.close(fig)


#def visualize_batches_intermediate_results(sketch, cam, batches):
def visualize_correspondences(
        sketch, cam, batches, selected_batches=[], plot_planes=False, VERBOSE=False):
    correspondences = []
    for batch_id, batch in enumerate(batches):
        if len(selected_batches) > 0 and not batch_id in selected_batches:
            continue
        for corr in batch["final_correspondences"]:
            correspondences.append([corr["stroke_id_0"], corr["stroke_id_1"], corr["symmetry_plane_id"]])

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.rcParams["figure.dpi"] = 200

    line_label_colors = sns.color_palette("Set1", n_colors=6)
    line_label_colors[5] = line_label_colors[4]
    cmap = sns.color_palette("tab20b", n_colors=len(sketch.strokes))
    cmap = sns.color_palette(list(sns.color_palette("Set1", n_colors=8))+
                             list(sns.color_palette("Dark2", n_colors=8))+
                             list(sns.color_palette("Accent", n_colors=8)),
                             n_colors=len(sketch.strokes))
    #cmap = sns.color_palette("Paired", n_colors=len(sketch.strokes))
    x_lim, y_lim = get_sketch_limits(sketch)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                        bottom=0.0, top=1.0)

    #fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
    #                    bottom=0.0, top=1.0)

    for j in range(3):
        sketch.display_strokes_2(fig=fig, ax=axes[j], color_process=lambda s: "#000000d8",
                                 linewidth_process=lambda s: 0.5)
        axes[j].set_xlim(x_lim)
        axes[j].set_ylim(y_lim)
        axes[j].set_aspect("equal")
        axes[j].axis("off")
    if plot_planes:
        batches_plane_points, _ = get_planes_3d(batches, selected_batches)
        for plane_points in batches_plane_points:
            if len(plane_points) == 0:
                continue

            for i in range(3):
                #patch = PolygonPatch(MultiPoint(cam.project_polyline(plane_points[i])).convex_hull,
                #                     fc=vp_colors[i], ec=vp_colors[i], 
                #                     alpha=0.4, zorder=0)
                #axes[i].add_patch(patch)
                plot_polygon(MultiPoint(cam.project_polyline(plane_points[i])).convex_hull, ax=axes[i], 
                    facecolor=vp_colors[i], edgecolor=vp_colors[i], alpha=0.4, zorder=0, add_points=False)

    self_sym_strokes = [[] for i in range(3)]

    axes[0].text(.95, .05, "Black lines are self-symmetric",
                 horizontalalignment='center',
                 transform=axes[0].transAxes)
    for axis_id in range(3):
        stroke_colors = ["black" for i in range(len(sketch.strokes))]
        graph = nx.Graph()
        for corr in correspondences:
            if corr[-1] != axis_id:
                continue
            graph.add_edge(corr[0], corr[1])
        components = [c for c in nx.connected_components(graph)]
        for c in components:
            if len(c) == 1:
                self_sym_strokes[axis_id].append(list(c)[0])
                color = "#000000"
                s_id = list(c)[0]
                stroke_colors[s_id] = color
                pts = np.array([p.coords for p in sketch.strokes[s_id].points_list])
                axes[axis_id].plot(pts[:, 0], pts[:, 1], color=color, lw=2)
                continue
            color = cmap[np.min(list(c))]
            for s_id in c:
                stroke_colors[s_id] = color
                pts = np.array([p.coords for p in sketch.strokes[s_id].points_list])
                axes[axis_id].plot(pts[:, 0], pts[:, 1], color=color, lw=3)
        axes[axis_id].text(.5, .95, "Symmetry correspondences for axis "+str(axis_id),
                              horizontalalignment='center',
                              transform=axes[axis_id].transAxes)
        add_sketch_vp_arrow(sketch, x_lim, y_lim, cam, axes[axis_id], axis_id, line_label_colors)
    if VERBOSE:
        plt.show()
    else:
        file_name = os.path.join(sketch.sketch_folder, "correspondences.png")
        if len(selected_batches) > 0:
            last_part = "correspondences_"+"_".join([str(sel) for sel in selected_batches])+".png"
            if os.path.exists(os.path.join(sketch.sketch_folder, "batches")):
                file_name = os.path.join(sketch.sketch_folder, "batches", last_part)
            else:
                file_name = os.path.join(sketch.sketch_folder, last_part)
        plt.savefig(file_name)

def add_sketch_vp_arrow(sketch, x_lim, y_lim, camera, axis, axis_id, colors):
    sketch_points = np.array([p.coords for s in sketch.strokes for p in s.points_list])
    lowest_point = sketch_points[np.argmax(sketch_points[:, 1])]

    if axis_id == 0:
        left_range = lowest_point[0] - x_lim[0]
        middle_left = np.array([lowest_point[0] - left_range/4, lowest_point[1]])
        vec = np.array(camera.vanishing_points_coords[0]) - middle_left
        vec /= np.linalg.norm(vec)
        axis.arrow(middle_left[0], middle_left[1], left_range/2*vec[0], left_range/2*vec[1],
                   width=5, color=colors[0], alpha=0.5)

    elif axis_id == 1:
        right_range = x_lim[1] - lowest_point[0]
        middle_right = np.array([lowest_point[0] + right_range/4, lowest_point[1]])
        vec = np.array(camera.vanishing_points_coords[1]) - middle_right
        vec /= np.linalg.norm(vec)
        axis.arrow(middle_right[0], middle_right[1], right_range/2*vec[0], right_range/2*vec[1],
                   width=5, color=colors[1], alpha=0.5)

    elif axis_id == 2:
        vec = np.array([0, 1])
        up_range = y_lim[1] - y_lim[0]
        middle_up = np.array([x_lim[1]-10, y_lim[0]+up_range/4])
        axis.arrow(middle_up[0], middle_up[1], up_range/2*vec[0], up_range/2*vec[1],
                   width=5, color=colors[2], alpha=0.5)

def get_planes_3d(batches, selected_batches=[]):
    batches_plane_points = []
    points = []
    for batch_id, batch in enumerate(batches):
        points += [inter[0] for inter in batch["intersections"]]
    points = np.array(points)
    bbox = tools_3d.bbox_from_points(points)
    for batch_id, batch in enumerate(batches):
        if len(selected_batches) > 0 and not batch_id in selected_batches:
            batches_plane_points += []    
            continue
        shifted_origins = [np.zeros(3) for i in range(3)]
        for i in range(3):
            shifted_origins[i][i] -= batch["symmetry_planes"][i]["signed_distance"]
        x_plane_points = [[shifted_origins[0][0], bbox[1], bbox[2]],
                          [shifted_origins[0][0], bbox[4], bbox[2]],
                          [shifted_origins[0][0], bbox[4], bbox[5]],
                          [shifted_origins[0][0], bbox[1], bbox[5]]]
        y_plane_points = [[bbox[0], shifted_origins[1][1], bbox[2]],
                          [bbox[3], shifted_origins[1][1], bbox[2]],
                          [bbox[3], shifted_origins[1][1], bbox[5]],
                          [bbox[0], shifted_origins[1][1], bbox[5]]]
        z_plane_points = [[bbox[0], bbox[1], shifted_origins[2][2]],
                          [bbox[3], bbox[1], shifted_origins[2][2]],
                          [bbox[3], bbox[4], shifted_origins[2][2]],
                          [bbox[0], bbox[4], shifted_origins[2][2]]]
        batches_plane_points.append([x_plane_points, y_plane_points, z_plane_points])
    
    plane_faces = [[0, 1, 2], [0, 2, 3]]
    return batches_plane_points, plane_faces

def plot_intersections(sketch, file_name="intersections.png", VERBOSE=False):
    fig, axes = get_sketch_fig(sketch)
    for inter in sketch.intersection_graph.get_intersections():
        if inter.is_tangential:
            axes.add_artist(Circle(xy=inter.inter_coords, radius=inter.acc_radius, color="red",
                                   ))
        elif inter.is_extended:
            axes.add_artist(Circle(xy=inter.inter_coords, radius=inter.acc_radius, color="green",
                                   ))
        elif hasattr(inter, "is_parallel") and inter.is_parallel:
            axes.add_artist(Circle(xy=inter.inter_coords, radius=inter.acc_radius, color="pink",
                                   ))
        else:
            axes.add_artist(Circle(xy=inter.inter_coords, radius=inter.acc_radius, color="blue"))
    sketch.display_strokes_2(fig=fig, ax=axes, norm_global=True,
        color_process=lambda s: "black", linewidth_data=lambda s:3)
    plot_end(sketch, fig, axes, file_name, VERBOSE)


def plot_candidate_correspondences(sketch, global_candidate_correspondences):

    tmp_per_axis_per_stroke_list = [[[] for i in range(len(sketch.strokes))] for i in range(3)]
    for cand in global_candidate_correspondences:
        tmp_per_axis_per_stroke_list[cand[4]][cand[0]].append(cand[1])
        tmp_per_axis_per_stroke_list[cand[4]][cand[1]].append(cand[0])
    candidate_correspondences_folder = os.path.join(sketch.sketch_folder, "candidate_correspondences")
    clean_folder(candidate_correspondences_folder)
    for axis in range(3):
        for s_id in range(len(sketch.strokes)):
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                bottom=0.0,
                                top=1.0)
            sketch.display_strokes_2(fig=fig, ax=axes, color_process=lambda s: "grey",
                                     norm_global=True)
            sketch.display_strokes_2(fig=fig, ax=axes, color_process=lambda s: "red",
                                     display_strokes=[s_id],
                                     norm_global=True,
                                     linewidth_data=lambda s: 3.0)
            if len(tmp_per_axis_per_stroke_list[axis][s_id]) > 0:
                sketch.display_strokes_2(fig=fig, ax=axes, color_process=lambda s: "green",
                                         display_strokes=tmp_per_axis_per_stroke_list[axis][s_id],
                                         norm_global=True,
                                         linewidth_data=lambda s: 3.0)
            axes.set_xlim(0, sketch.width)
            axes.set_ylim(sketch.height, 0)
            axes.axis("equal")
            axes.axis("off")
            plt.savefig(os.path.join(candidate_correspondences_folder, "axis_"+str(axis)+"_stroke_"+str(np.char.zfill(str(s_id), 3))+".png"))
