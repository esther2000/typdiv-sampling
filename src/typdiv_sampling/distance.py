from collections import defaultdict

import numpy as np


def dist(p1, p2):
    out = np.linalg.norm(p1 - p2)
    return out


def get_dist_matrix(data):
    n = len(data)
    out = []
    for i in range(n):
        dists = []
        for j in range(n):
            dists.append(dist(data[i], data[j]))
        out.append(dists)

    return np.array(out)


def dist_score(sol_idxs, d):
    return d[sol_idxs, :].T[sol_idxs, :].T.sum() / 2


def get_summed_dist_dict(distance_dict, all_langs, current_langs):
    summed_dist_dict = defaultdict(int)

    for l1 in current_langs:
        for l2 in all_langs:
            summed_dist_dict[l2] += distance_dict[l1][l2]

    return summed_dist_dict


def get_first_point(dists):
    return np.argmax(dists.sum(axis=1))
