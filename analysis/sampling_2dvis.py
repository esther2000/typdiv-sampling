"""
Description:    Visualize sampling algorithms in 2D space
Note that sample_mdp and sample_mmdp are duplicated because there are output format differences
"""

import sys
sys.path.insert(0, "../")

import argparse
import pandas as pd
import random
from typdiv.typ_dist import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


RAND_SEED = 123
random.seed(RAND_SEED)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sampling_method",
        type=str,
        default="typ_mdp",
        help="Sampling method, choose from: random_family, random_genus, typ_mdp, typ_mmdp", # TODO: add random?
    )
    parser.add_argument(
        "-k",
        "--k_langs",
        type=int,
        default=50,
        help="Number of languages to select from selection.",
    )
    return parser.parse_args()


def sample_mdp(max_langs, dist_mtx):
    """
    Maximum Diversity Problem
    Sample k languages from N, where we iteratively add the
    next point that yields the largest summed distance (greedy).
    """

    all_langs = [i for i in range(len(dist_mtx))]
    most_distant_langid = dist_mtx.sum(axis=1).argmax()

    langs = [most_distant_langid]
    all_langs.remove(most_distant_langid)
    while len(langs) <= max_langs-1:
        summed_dist = get_summed_dist_dict(dist_mtx, all_langs, langs)
        next_most_distant = max(summed_dist, key=lambda x: summed_dist[x])
        all_langs.remove(next_most_distant)
        langs.append(next_most_distant)

    return langs


def sample_mmdp(max_langs, dist_mtx):
    """
    MaxMin Diversity Problem
    Sample k languages from N, where we iteratively add the
    next point that yields the maximum minimum distance between
    any two points in k.
    """

    p1 = get_first_point(dist_mtx)
    p2 = dist_mtx[p1].argmax()

    L = { i for i in range(dist_mtx.shape[0]) }
    S = {p1, p2}

    while len(S) < max_langs:
        rest_L = tuple(L.symmetric_difference(S))
        rest_dists = dist_mtx[rest_L,:].T[tuple(S),:].T
        S.add(rest_L[rest_dists.min(axis=1).argmax()])

    return list(S)


def sample_random_family(languages, max_langs):
    """
    Sample k languages from N where we sample from
    language families as uniformly as the data allows.
    """
    g_df = pd.read_csv("sources/grambank-v1.0.3/cldf/languages.csv")
    g_df = g_df[g_df["Glottocode"].isin(languages)]
    # Sample one language from each family
    codes = (
        g_df.groupby("Family_name")
        .apply(lambda x: x.sample(1, random_state=RAND_SEED))
        .reset_index(drop=True)["Glottocode"]
    )

    # We're lucky, n_families is the same as the number of langs we're looking for
    if len(codes) == max_langs:
        return codes

    # We have more families than we need, sample what we need
    if len(codes) > max_langs:
        return random.sample(codes.tolist(), max_langs)

    # Figure out how many to get from the families (as even per lang as possible)
    n_families = len(g_df["Family_name"].unique())
    s_size = max_langs // n_families
    codes = (
        g_df.groupby("Family_name")
        .apply(lambda x: x.sample(s_size, random_state=RAND_SEED))
        .reset_index(drop=True)["Glottocode"]
        .tolist()
    )
    # Fill up the remaining langs by random selection from all families
    while len(codes) < max_langs:
        new_sample = g_df[~(g_df["Glottocode"].isin(codes))].sample(
            1, random_state=RAND_SEED
        )
        codes.append(new_sample["Glottocode"].values[0])

    return codes


def main():

    args = create_arg_parser()

    V = np.load('../data/normal_2000_loc1_scale10.npy')
    all_langs = [i for i in range(len(V))]
    dist_dict = get_dist_matrix(V)

    # sample
    if args.sampling_method == 'typ_mdp':
        sample = sample_mdp(args.k_langs, dist_dict)
    elif args.sampling_method == 'typ_mmdp':
        sample = sample_mmdp(args.k_langs, dist_dict)
    else:
        print('Error: Unknown sampling method.')

    print(dist_score(sample, dist_dict))
    print(sample)

    # visualize
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    ax.scatter(V[all_langs, 0], V[all_langs, 1], c='#3935db')#, alpha=0.5)
    ax.scatter(V[sample, 0], V[sample, 1], c='#db3535')

    # remove axes
    ax.set(xticklabels=[], yticklabels=[])
    ax.tick_params(bottom=False, left=False)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
