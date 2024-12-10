from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances


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


def make_language_distances(feature_dataframe, normalize: bool = True):
    """Create distance matrix from a language feature dataframe. Returns the distance matrix as a dataframe."""

    sim_matrix = nan_euclidean_distances(feature_dataframe, feature_dataframe)
    glottocodes = feature_dataframe.index
    # NOTE: fillna(0) for maximisation only!
    sim_df = pd.DataFrame(sim_matrix, columns=glottocodes, index=glottocodes).fillna(0)
    if normalize:
        sim_df = (sim_df - sim_df.min().min()) / (
            sim_df.max().max() - sim_df.min().min()
        )

    return sim_df
