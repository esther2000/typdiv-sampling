"""
Description:    Visualize sampling algorithms in 2D space
Note that sample_maxsum and sample_maxmin are duplicated because there are output format differences
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typdiv_sampling.distance import (
    dist_score,
    get_dist_matrix,
    get_first_point,
    get_summed_dist_dict,
)
from typdiv_sampling.constants import DATA_PATH

RAND_SEED = 123
random.seed(RAND_SEED)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sampling_method",
        type=str,
        default="typ_maxsum",
        help="Sampling method, choose from: typ_maxsum, typ_maxmin",
    )
    parser.add_argument(
        "-k",
        "--k_langs",
        type=int,
        default=50,
        help="Number of languages to select from selection.",
    )
    return parser.parse_args()


def sample_maxsum(max_langs, dist_mtx):
    """
    Maximum Diversity Problem
    Sample k languages from N, where we iteratively add the
    next point that yields the largest summed distance (greedy).
    """

    all_langs = [i for i in range(len(dist_mtx))]
    most_distant_langid = dist_mtx.sum(axis=1).argmax()

    langs = [most_distant_langid]
    all_langs.remove(most_distant_langid)
    while len(langs) <= max_langs - 1:
        summed_dist = get_summed_dist_dict(dist_mtx, all_langs, langs)
        next_most_distant = max(summed_dist, key=lambda x: summed_dist[x])
        all_langs.remove(next_most_distant)
        langs.append(next_most_distant)

    return langs


def sample_maxmin(max_langs, dist_mtx):
    """
    MaxMin Diversity Problem
    Sample k languages from N, where we iteratively add the
    next point that yields the maximum minimum distance between
    any two points in k.
    """

    p1 = get_first_point(dist_mtx)
    p2 = dist_mtx[p1].argmax()

    L = {i for i in range(dist_mtx.shape[0])}
    S = {p1, p2}

    while len(S) < max_langs:
        rest_L = tuple(L.symmetric_difference(S))
        rest_dists = dist_mtx[rest_L, :].T[tuple(S), :].T
        S.add(rest_L[rest_dists.min(axis=1).argmax()])

    return list(S)


def main():
    args = create_arg_parser()

    V = np.load(DATA_PATH / "normal_2000_loc1_scale10.npy")
    all_langs = [i for i in range(len(V))]
    dist_dict = get_dist_matrix(V)

    if args.sampling_method == "typ_maxsum":
        sample = sample_maxsum(args.k_langs, dist_dict)
    elif args.sampling_method == "typ_maxmin":
        sample = sample_maxmin(args.k_langs, dist_dict)
    else:
        raise ValueError("Error: Unknown sampling method.")

    print(dist_score(sample, dist_dict))
    print(sample)

    # visualize
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    ax.scatter(V[all_langs, 0], V[all_langs, 1], c="#3935db")  # , alpha=0.5)
    ax.scatter(V[sample, 0], V[sample, 1], c="#db3535")

    # remove axes
    ax.set(xticklabels=[], yticklabels=[])
    ax.tick_params(bottom=False, left=False)

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
