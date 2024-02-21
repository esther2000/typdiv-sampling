#!/usr/bin/python3
"""
Description:    ...
Usage:          ...
"""

import argparse
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--gb_folder", type=str,
                        default='sources/grambank-v1.0.3',
                        help="Folder where Grambank download is stored")
    args = parser.parse_args()
    return args


def main():

    args = create_arg_parser()
    # TODO: add weighted similarity: rare features, transfer learning relevance

    # Load Grambank matrix
    gb_matrix = pd.read_csv('gb_lang_feat_vals.csv')
    gb_feats = [x for x in gb_matrix.columns.to_list() if x.startswith('GB')]

    # Make vector per language
    lang_vecs = {row["Lang_ID"]: row[[x for x in gb_feats]].to_list() for _, row in gb_matrix.iterrows()}
    for lang, vec in lang_vecs.items():
        lang_vecs[lang] = [x if x != 'no_cov' and x!='?' else float("NaN") for x in vec]

    # Compute similarity matrix
    langs = gb_matrix['Lang_ID'].to_list()
    vecs = [lang_vecs[lang] for lang in langs]
    sim_matrix = nan_euclidean_distances(vecs, vecs)

    # Write to file
    sim_df = pd.DataFrame(sim_matrix, columns=langs, index=langs).fillna(0)
    sim_df.to_csv('gb_vec_sim_0.csv')


if __name__ == '__main__':
    main()
