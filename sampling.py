"""
Description:    sample k languages from N, using different methods
Usage:          python sampling.py TODO: add options
"""


import argparse

import numpy as np
import pandas as pd
import random
from typ_dist import *

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
        help="Sampling method, choose from: random, random_family, random_genus, typ_mdp, typ_mmdp",
    )
    parser.add_argument(
        "--k_langs",
        type=int,
        default=10,
        help="Number of languages to select from selection.",
    )
    parser.add_argument(
        "--n_langs_file",
        type=str,
        default='langs_gb.txt',
        help="File with languages to select from, one language code per line.",  # TODO: glotto for now...
    )

    return parser.parse_args()


def sample_mdp(all_langs, max_langs, dist_mtx, id2lang):
    """
    TODO: add proper description
    mdp, fpf
    """

    all_langs = [i for i in range(len(all_langs))]
    most_distant_langid = dist_mtx.sum(axis=1).argmax()

    langs = [most_distant_langid]
    all_langs.remove(most_distant_langid)
    while len(langs) <= max_langs-1:
        summed_dist = get_summed_dist_dict(dist_mtx, all_langs, langs)
        next_most_distant = max(summed_dist, key=lambda x: summed_dist[x])
        all_langs.remove(next_most_distant)
        langs.append(next_most_distant)

    return [id2lang[i] for i in langs]


def sample_mmdp(max_langs, dist_mtx, id2lang):
    """
    TODO: add proper description
    maxmindivp
    """

    # TODO: all_langs --> dist mtx should be cropped to these... (maybe in main)

    p1 = get_first_point(dist_mtx)
    p2 = np.nanargmax(dist_mtx[p1])

    L = { i for i in range(dist_mtx.shape[0]) }
    S = {p1, p2}

    while len(S) < max_langs:
        rest_L = tuple(L.symmetric_difference(S))
        rest_dists = dist_mtx[rest_L,:].T[tuple(S),:].T
        S.add(rest_L[np.nanargmax(np.nanmin(rest_dists, axis=1))])

    return [id2lang[i] for i in S]


def sample_random_family(languages, max_langs):
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
    n_families = len(g_df["Family_ID"].unique())
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
        codes.append(new_sample["Glottocode"])

    return codes


def sample_random_genus(languages, max_langs):
    g_df = pd.read_csv("data/wals_dedup.csv")
    g_df = g_df[g_df["Glottocode"].isin(languages)]
    # Sample one language from each family
    codes = (
        g_df.groupby("Genus")
        .apply(lambda x: x.sample(1, random_state=RAND_SEED))
        .reset_index(drop=True)["Glottocode"]
    )

    # We're lucky, n_genera is the same as the number of langs we're looking for
    if len(codes) == max_langs:
        return codes

    # We have more genera than we need, sample what we need
    if len(codes) > max_langs:
        return random.sample(codes.tolist(), max_langs)

    # Figure out how many to get from the genera (as even per lang as possible)
    n_genera = len(g_df["Genus"].unique())
    s_size = max_langs // n_genera
    codes = (
        g_df.groupby("Genus")
        .apply(lambda x: x.sample(s_size, random_state=RAND_SEED))
        .reset_index(drop=True)["Glottocode"]
        .tolist()
    )
    # Fill up the remaining langs by random selection from all genera
    while len(codes) < max_langs:
        new_sample = g_df[~(g_df["Glottocode"].isin(codes))].sample(
            1, random_state=RAND_SEED
        )
        codes.append(new_sample["Glottocode"])


def main():

    args = create_arg_parser()

    with open('data/gb_langs.txt') as langsfile:
        n_langs = [x.strip() for x in langsfile.readlines()]

    gb_matrix = pd.read_csv('data/gb_vec_sim.csv', index_col=0)

    id2lang = gb_matrix.columns.tolist()
    dist_mtx = gb_matrix.to_numpy()

    print(f"Random within language families: {', '.join(sorted(sample_random_family(n_langs, args.k_langs)))}")
    print(f"Random within language genera: {', '.join(sorted(sample_random_genus(n_langs, args.k_langs)))}")
    print(f"MDP typological sampling: {', '.join(sorted(sample_mdp(n_langs, args.k_langs, dist_mtx, id2lang)))}")
    print(f"MMDP typological sampling: {', '.join(sample_mmdp(args.k_langs, dist_mtx, id2lang))}")

    # TODO: get k (num) and N (langs), verify
    # TODO: get sampling method, if typological --> run distance calculation first (with weights) --> or: do we do this in preprocessing.py? perhaps... and arguments could be given there, too
    # TODO: sample
    # TODO: output sample

    # TODO: how to deal with visualization in 2D space? add separate option?
    #  (I think it should be a separate program actually, because it requires N as number etc. (although..... idk))


if __name__ == '__main__':
    main()
