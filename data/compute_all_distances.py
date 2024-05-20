#!/usr/bin/python3
"""
Description:    Calculate distances between pairs of languages based on Grambank feature values
Usage:          python compute_all_distances.py -g <GRAMBANK_FOLDER> -o <OUTPUT_FILE> -b (optional)

TODO: add weighted similarity: rare features, transfer learning relevance
"""

import argparse

import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import MinMaxScaler


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gb_folder",
        type=str,
        default="sources/grambank-v1.0.3",
        help="Folder where Grambank download is stored",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="gb_vec_sim.csv",
        help="Name of file that output distances should be written to",
    )
    parser.add_argument(
        "-b",
        "--binarize",
        action="store_true",
        help="Option for binarizing multi-value features",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Option for normalizing distances (min-max)",
    )
    args = parser.parse_args()
    return args


def binarize(gb_df, mv_feats):
    """Binarize multi-value features"""
    for feat in mv_feats.keys():
        for val in range(0, mv_feats[feat]):
            # binarize multi-value features
            gb_df[f"{feat}_{str(val)}"] = (
                (gb_df[f"{feat}"] == str(val)).astype(int).astype(str)
            )
            # if the original value was "?", put it back
            gb_df.loc[gb_df[f"{feat}"] == "?", f"{feat}_{str(val)}"] = "?"

    # remove original multi-value feature columns
    gb_df = gb_df.drop(columns=mv_feats.keys())

    return gb_df, gb_df.columns.to_list()[2:]


def main():
    args = create_arg_parser()
    mv_feats = {
        "GB024": 3,
        "GB025": 3,
        "GB065": 3,
        "GB130": 3,
        "GB193": 4,
        "GB203": 4,
    }  # IDs and num possible values

    # Load Grambank matrix
    gb_matrix = pd.read_csv("gb_lang_feat_vals.csv")
    gb_feats = [x for x in gb_matrix.columns.to_list() if x.startswith("GB")]

    # Optional: binarize multi-value features
    if args.binarize:
        gb_matrix, gb_feats = binarize(gb_matrix, mv_feats)

    # Make vector per language
    lang_vecs = {
        row["Lang_ID"]: row[[x for x in gb_feats]].to_list()
        for _, row in gb_matrix.iterrows()
    }
    for lang, vec in lang_vecs.items():
        lang_vecs[lang] = [
            x if x != "no_cov" and x != "?" else float("NaN") for x in vec
        ]

    # Compute similarity matrix
    langs = gb_matrix["Lang_ID"].to_list()
    vecs = [lang_vecs[lang] for lang in langs]
    sim_matrix = nan_euclidean_distances(vecs, vecs)

    # Write to file
    sim_df = pd.DataFrame(sim_matrix, columns=langs, index=langs).fillna(0)
    if args.normalize:
        scaler = MinMaxScaler()
        norm_data = scaler.fit_transform(sim_df)
        sim_df = pd.DataFrame(norm_data, columns=sim_df.columns, index=langs).fillna(0)

    sim_df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
