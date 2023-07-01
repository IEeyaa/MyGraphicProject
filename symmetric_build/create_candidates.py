from copy import deepcopy

import numpy as np
from scipy.spatial.distance import directed_hausdorff

# 生成最终的candidates
from sketch.sketch_info import Candidates


def generate_final_candidates(candidates, stroke_number, group_infor):
    plane_number = np.max(np.array([candidate[4] for candidate in candidates])) + 1
    stroke_plane_candidates = [
        [[] for j in range(plane_number)]
        for i in range(stroke_number + 1)
    ]
    for index, item in enumerate(candidates):
        first_candidate_id = len(stroke_plane_candidates[item[0]][item[4]])
        snd_candidate_id = len(stroke_plane_candidates[item[1]][item[4]])
        # first-stroke
        proxy_distances = []
        for p_id in item[5]:
            p_d = max(directed_hausdorff(np.array(item[2]), np.array(group_infor[item[0]][p_id]))[0],
                      directed_hausdorff(np.array(group_infor[item[0]][p_id]), np.array(item[2]))[0])
            proxy_distances.append(p_d)
        stroke_plane_candidates[item[0]][item[4]].append(Candidates(index, item[2],
                                                                     [item[0], item[1]],
                                                                     [item[7], item[8]],
                                                                     [first_candidate_id, snd_candidate_id],
                                                                     [item[5], deepcopy(proxy_distances)]
                                                                     ))
        if item[0] != item[1]:
            # second_stroke
            proxy_distances = []
            for p_id in item[6]:
                p_d = max(directed_hausdorff(np.array(item[3]), np.array(group_infor[item[1]][p_id]))[0],
                          directed_hausdorff(np.array(group_infor[item[1]][p_id]), np.array(item[3]))[0])
                proxy_distances.append(p_d)
            stroke_plane_candidates[item[1]][item[4]].append(Candidates(index, item[3],
                                                                         [item[1], item[0]],
                                                                         [-1, -1],
                                                                         [snd_candidate_id, first_candidate_id],
                                                                         [item[6], deepcopy(proxy_distances)]
                                                                         ))
    return stroke_plane_candidates, plane_number
