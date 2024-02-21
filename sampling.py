"""
Description:    Sample k languages from N, using different methods
Usage:          python sampling.py -s <SAMPLING_METHOD> -k <NUM_LANGS> -n <ALL_LANGS> -l <LANG_SIM>
"""


import argparse
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
        help="Sampling method, choose from: random_family, random_genus, typ_mdp, typ_mmdp", # TODO: add random?
    )
    parser.add_argument(
        "-k",
        "--k_langs",
        type=int,
        default=10,
        help="Number of languages to select from selection.",
    )
    parser.add_argument(
        "-n",
        "--frame",
        type=str,
        default='data/frames/langs_gb.txt',
        help="File with languages to select from, one Glottocode per line.",
    )
    parser.add_argument(
        "-l",
        "--language_distances",
        type=str,
        default='data/gb_vec_sim_0.csv',
        help="File with pairwise language distances.",
    )

    return parser.parse_args()


def sample_mdp(max_langs, dist_mtx, id2lang):
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

    return [id2lang[i] for i in langs]


def sample_mmdp(max_langs, dist_mtx, id2lang):
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

    return [id2lang[i] for i in S]


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
    """
    Sample k languages from N where we sample from
    language genera as uniformly as the data allows.
    """
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

    return codes


def main():

    args = create_arg_parser()

    # define sampling frame
    with open(args.frame) as all_langs:
        n_langs = [x.strip() for x in all_langs.readlines()]

    # load typological distances
    if args.sampling_method == 'typ_mdp' or args.sampling_method == 'typ_mmdp':
        dist_df = pd.read_csv(args.language_distances, index_col=0)
        dist_df = dist_df[n_langs].loc[n_langs]
        dist_mtx = dist_df.to_numpy()
        id2lang = dist_df.columns.tolist()

    # sample
    if args.sampling_method == 'random_family':
        sample = sample_random_family(n_langs, args.k_langs)
    elif args.sampling_method == 'random_genus':
        sample = sample_random_genus(n_langs, args.k_langs)
    elif args.sampling_method == 'typ_mdp':
        sample = sample_mdp(args.k_langs, dist_mtx, id2lang)
    elif args.sampling_method == 'typ_mmdp':
        sample = sample_mmdp(args.k_langs, dist_mtx, id2lang)
    else:
        print('Error: Unknown sampling method.')

    # write sample to outfile
    outfile = f'evaluation/samples/{args.sampling_method}-{args.frame.split("/")[-1][:-4]}-{args.k_langs}' \
              f'-{args.language_distances.split("/")[-1][:-4]}-{RAND_SEED}.txt'

    with open(outfile, 'w') as of:
        for l in sample:
            of.write(l + '\n')


if __name__ == '__main__':
    main()
