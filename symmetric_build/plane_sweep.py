import numpy as np
from copy import deepcopy
from scipy.spatial import distance
from skspatial.objects import Plane

from symmetric_build.common_tools import copy_correspondences_batch
from sketch.intersections import get_intersections_simple_batch, get_line_coverages_simple, relift_intersections
from symmetric_build.ortools_models import solve_symm_bip_ortools
from tools import tools_3d
from tools.tools_cluster import cluster_3d_lines_correspondence

SKETCH_COMPACITY = False
INTERSECTION_CONSTRAINT = True
SKETCH_CONNECTIVITY = True
LINE_COVERAGE = True
ABLATE_LINE_COVERAGE = False
ABLATE_PROXIMITY = False
ABLATE_CURVE_CORR = False
ABLATE_ECCENTRICITY = False
ABLATE_TANGENTIAL_INTERSECTIONS = False
ABLATE_ANCHORING = False
selected_planes = [0, 1, 2]


def plane_sweep(sketch, cam, batch_id, batch, symm_candidates,
                fixed_strokes, fixed_intersections, fixed_planes_scale_factors,
                fixed_line_coverages,
                planes_scale_factors,
                planes_combs, main_axis,
                extreme_intersections_distances_per_stroke, per_stroke_triple_intersections, accumulated_obj_value,
                batches_results, solver="ortools"):
    global solve_symm
    best_comb = -1
    best_obj_value = -10000.0
    solve_symm = solve_symm_bip_ortools
    for planes_comb_id, planes_comb in enumerate(planes_combs):
        print(str(planes_comb_id) + "/" + str(len(planes_combs)))

        refl_mats = []

        planes_point_normal = []
        if batch_id > 0 or np.sum([len(fixed_strokes[i]) for i in range(len(fixed_strokes))]) > 0:
            for i in range(3):
                focus_vp = i
                sym_plane_point = np.zeros(3, dtype=np.float_)
                sym_plane_normal = np.zeros(3, dtype=np.float_)
                sym_plane_normal[focus_vp] = 1.0
                sym_plane_point = cam.cam_pos + \
                                  planes_scale_factors[i][planes_comb[i]] * (np.array(sym_plane_point) - cam.cam_pos)
                planes_point_normal.append([sym_plane_point, sym_plane_normal])
            for p, n in planes_point_normal:
                refl_mat = tools_3d.get_reflection_mat(p, n)
                refl_mats.append(refl_mat)
        local_planes_scale_factors = [planes_scale_factors[0][planes_comb[0]],
                                      planes_scale_factors[1][planes_comb[1]],
                                      planes_scale_factors[2][planes_comb[2]]]
        all_planes_scale_factors = deepcopy(fixed_planes_scale_factors)
        all_planes_scale_factors.append(local_planes_scale_factors)
        local_candidate_correspondences, correspondence_ids = copy_correspondences_batch(
            symm_candidates, batch, fixed_strokes, refl_mats,
            local_planes_scale_factors,
            cam, sketch)

        if len(local_candidate_correspondences) == 0:
            continue

        per_stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
        cluster_3d_lines_correspondence(local_candidate_correspondences,
                                        per_stroke_proxies, sketch)
        eccentricity_weights = [[] for i in range(len(sketch.strokes))]

        intersections_3d_simple = get_intersections_simple_batch(
            per_stroke_proxies, sketch, cam, batch, fixed_strokes)

        if batch_id == 0:
            min_dists = []
            epsilons = []
            for inter in intersections_3d_simple:
                first_cam_depths = inter.cam_depths[0]
                if len(first_cam_depths) == 0:
                    first_cam_depths = [inter.fix_depth]
                snd_cam_depths = inter.cam_depths[1]
                if len(snd_cam_depths) == 0:
                    snd_cam_depths = [inter.fix_depth]
                min_dists.append(np.min(distance.cdist(np.array(first_cam_depths).reshape(-1, 1),
                                                       np.array(snd_cam_depths).reshape(-1, 1))))
                epsilons.append(inter.epsilon)

        line_coverages_simple = []
        if LINE_COVERAGE:
            line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                              extreme_intersections_distances_per_stroke)

        obj_value, final_correspondences, final_proxis, final_intersections, final_line_coverages, \
        final_end_intersections, solution_terms, final_full_anchored_stroke_ids, final_half_anchored_stroke_ids, final_triple_intersections = \
            solve_symm(
                local_candidate_correspondences, per_stroke_proxies, intersections_3d_simple, line_coverages_simple,
                batch, fixed_strokes, fixed_intersections, sketch,
                return_final_strokes=True,
                line_coverage=LINE_COVERAGE, eccentricity_weights=eccentricity_weights,
                best_obj_value=best_obj_value - 1,
                per_stroke_triple_intersections=per_stroke_triple_intersections,
                main_axis=main_axis,
                ABLATE_LINE_COVERAGE=ABLATE_LINE_COVERAGE,
                ABLATE_PROXIMITY=ABLATE_PROXIMITY,
                ABLATE_ECCENTRICITY=ABLATE_ECCENTRICITY,
                ABLATE_TANGENTIAL_INTERSECTIONS=ABLATE_TANGENTIAL_INTERSECTIONS,
                ABLATE_ANCHORING=ABLATE_ANCHORING
            )

        if np.isclose(obj_value, -1.0):
            continue

        if obj_value > best_obj_value:
            best_obj_value = obj_value
            best_comb = planes_comb

    accumulated_obj_value.append(best_obj_value)

    if best_comb == -1:
        fixed_planes_scale_factors.append([])
        return []

    # output best alignment
    best_planes = []
    best_planes_point_normal = []

    for i in selected_planes:
        focus_vp = i
        sym_plane_point = np.zeros(3, dtype=np.float_)
        sym_plane_normal = np.zeros(3, dtype=np.float_)
        sym_plane_normal[focus_vp] = 1.0
        sym_plane_point = cam.cam_pos + \
                          planes_scale_factors[selected_planes[i]][best_comb[selected_planes[i]]] * (
                                  np.array(sym_plane_point) - cam.cam_pos)
        best_planes.append(Plane(sym_plane_point, sym_plane_normal))
        best_planes_point_normal.append([sym_plane_point, sym_plane_normal])

    refl_mats = []
    for p, n in best_planes_point_normal:
        refl_mat = tools_3d.get_reflection_mat(p, n)
        refl_mats.append(refl_mat)
    local_candidate_correspondences, correspondence_ids = copy_correspondences_batch(
        symm_candidates, batch, fixed_strokes, refl_mats,
        [planes_scale_factors[0][best_comb[0]],
         planes_scale_factors[1][best_comb[1]],
         planes_scale_factors[2][best_comb[2]]],
        cam, sketch)

    per_stroke_proxies = [[] for s_id in range(len(sketch.strokes))]
    cluster_3d_lines_correspondence(local_candidate_correspondences, per_stroke_proxies, sketch)

    eccentricity_weights = [[] for i in range(len(sketch.strokes))]

    intersections_3d_simple = get_intersections_simple_batch(per_stroke_proxies,
                                                             sketch, cam, batch, fixed_strokes)

    if batch_id == 0:
        min_dists = []
        for inter in intersections_3d_simple:
            first_cam_depths = inter.cam_depths[0]
            if len(first_cam_depths) == 0:
                first_cam_depths = [inter.fix_depth]
            snd_cam_depths = inter.cam_depths[1]
            if len(snd_cam_depths) == 0:
                snd_cam_depths = [inter.fix_depth]
            min_dists.append(np.min(distance.cdist(np.array(first_cam_depths).reshape(-1, 1),
                                                   np.array(snd_cam_depths).reshape(-1, 1))))

    line_coverages_simple = get_line_coverages_simple(intersections_3d_simple, sketch,
                                                      extreme_intersections_distances_per_stroke)

    obj_value, final_correspondences, final_proxis, final_intersections, final_line_coverages, \
    final_end_intersections, solution_terms, final_full_anchored_stroke_ids, final_half_anchored_stroke_ids, final_triple_intersections = \
        solve_symm(
            local_candidate_correspondences, per_stroke_proxies, intersections_3d_simple, line_coverages_simple,
            batch, fixed_strokes, fixed_intersections, sketch,
            return_final_strokes=True,
            line_coverage=LINE_COVERAGE, eccentricity_weights=eccentricity_weights,
            best_obj_value=best_obj_value - 1, per_stroke_triple_intersections=per_stroke_triple_intersections,
            main_axis=main_axis,
            ABLATE_LINE_COVERAGE=ABLATE_LINE_COVERAGE,
            ABLATE_PROXIMITY=ABLATE_PROXIMITY,
            ABLATE_ECCENTRICITY=ABLATE_ECCENTRICITY,
            ABLATE_TANGENTIAL_INTERSECTIONS=ABLATE_TANGENTIAL_INTERSECTIONS,
            ABLATE_ANCHORING=ABLATE_ANCHORING
        )

    if LINE_COVERAGE:
        for s_i, final_line_coverage in enumerate(final_line_coverages):
            if final_line_coverage > 0.0:
                fixed_line_coverages[s_i] = final_line_coverage
    results = {"final_proxies": []}
    for s_id, s in enumerate(sketch.strokes[:batch[1] + 1]):
        if final_proxis[s_id] is not None:
            results["final_proxies"].append(final_proxis[s_id].tolist())
        elif len(fixed_strokes[s_id]) > 0:
            results["final_proxies"].append(fixed_strokes[s_id].tolist())
        else:
            results["final_proxies"].append([])
    results["ref_final_proxies"] = []
    for inter in final_intersections:
        fixed_intersections.append(inter[2])

    results["symmetry_planes"] = [{"plane_normal": list(plane.normal),
                                   "signed_distance": plane.distance_point_signed([0, 0, 0]),
                                   }
                                  for plane_id, plane in enumerate(best_planes)]
    results["symmetry_correspondences"] = [{"stroke_id_0": int(corr[0]),
                                            "stroke_id_1": int(corr[1]),
                                            "stroke_3d_0": corr[2].tolist(),
                                            "stroke_3d_1": corr[3].tolist(),
                                            "symmetry_plane_id": corr[4],
                                            "self_sym_inter_id_0": int(corr[5]),
                                            "self_sym_inter_id_1": int(corr[6])
                                            }
                                           for corr in final_correspondences]
    results["local_correspondences"] = [{"stroke_id_0": int(corr[0]),
                                         "stroke_id_1": int(corr[1]),
                                         "stroke_3d_0": corr[2].tolist(),
                                         "stroke_3d_1": corr[3].tolist(),
                                         "symmetry_plane_id": corr[4],
                                         # get connectivity information in cas of self-symmetric strokes
                                         "self_sym_inter_id_0": int(corr[5][0]) if type(corr[5]) == list else corr[5],
                                         "self_sym_inter_id_1": int(corr[6][0]) if type(corr[6]) == list else corr[6]
                                         }
                                        for corr in local_candidate_correspondences]
    results["final_correspondences"] = [{"stroke_id_0": int(corr[0]),
                                         "stroke_id_1": int(corr[1]),
                                         "stroke_3d_0": corr[2].tolist(),
                                         "stroke_3d_1": corr[3].tolist(),
                                         "symmetry_plane_id": int(corr[4]),
                                         # get connectivity information in cas of self-symmetric strokes
                                         "self_sym_inter_id_0": int(corr[5][0]) if type(corr[5]) == list else int(
                                             corr[5]),
                                         "self_sym_inter_id_1": int(corr[6][0]) if type(corr[6]) == list else int(
                                             corr[6])
                                         }
                                        for corr in final_correspondences]

    results["batch_indices"] = [int(batch[0]), int(batch[1])]
    results["fixed_strokes"] = [s.tolist() if len(s) > 0 else [] for s in fixed_strokes]
    results["intersections"] = relift_intersections(
        final_intersections, fixed_strokes, final_proxis, sketch, cam)
    batches_results.append(results)

    for s_id, proxy in enumerate(final_proxis):
        if proxy is None:
            continue
        if s_id <= batch[1]:
            fixed_strokes[s_id] = proxy

    fixed_planes_scale_factors.append([planes_scale_factors[0][best_comb[0]],
                                       planes_scale_factors[1][best_comb[1]],
                                       planes_scale_factors[2][best_comb[2]]])
