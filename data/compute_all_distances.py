#!/usr/bin/python3
"""
Description:    Calculate distances between pairs of languages based on Grambank feature values
Usage:          python compute_all_distances.py -g <GRAMBANK_FOLDER> -o <OUTPUT_FILE> -b (optional)
"""

import argparse

import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
from pathlib import Path

# location of this file, so it does not matter from where this script is called
# TODO: move this to a constants.py or something since it's copy pasted atm
CWD = Path(__file__).parent
PROJECT_ROOT = CWD.parent
DATA = PROJECT_ROOT / "data"


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dist_file",
        type=Path,
        default=DATA / "gb_vec_sim.csv",
        help="Name of file that output distances should be written to",
    )
    parser.add_argument(
        "-b",
        "--binarize",
        action="store_true",
        help="Option for binarizing multi-value features",
    )
    parser.add_argument(
        "-d",
        "--data_output_file",
        type=Path,
        default=DATA / "gb_binarized.csv",
        help="Name of file that binarized grambank version should be written to",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Option for normalizing distances (min-max)",
    )
    parser.add_argument(
        "-c",
        "--crop_perc",
        type=float,
        help="Option: specify minimum feature coverage per language",
    )
    parser.add_argument(
        "-g",
        "--gb_file",
        default=DATA / "gb_lang_feat_vals.csv",
        help="Path to Grambank file with features per language.",
    )
    args = parser.parse_args()
    return args


def binarize(gb_df, mv_feats):
    """Binarize multi-value features
    TODO: this could probably be done more elegantly but it works"""
    for feat in mv_feats:
        gb_df[f"{feat}_1"] = (
            (gb_df[f"{feat}"] == "1").astype(int).astype(str)
        )  # label 1
        gb_df[f"{feat}_2"] = (
            (gb_df[f"{feat}"] == "2").astype(int).astype(str)
        )  # label 2

        # label 3 (both)
        gb_df.loc[gb_df[f"{feat}"] == "3", f"{feat}_1"] = "1"
        gb_df.loc[gb_df[f"{feat}"] == "3", f"{feat}_2"] = "1"

        # label 0 (none)
        gb_df.loc[gb_df[f"{feat}"] == "0", f"{feat}_1"] = "0"
        gb_df.loc[gb_df[f"{feat}"] == "0", f"{feat}_2"] = "0"

        # if original value was '?', put this back
        gb_df.loc[gb_df[f"{feat}"] == "?", f"{feat}_1"] = "?"
        gb_df.loc[gb_df[f"{feat}"] == "?", f"{feat}_2"] = "?"

        # if original value was 'no_cov', put this back
        gb_df.loc[gb_df[f"{feat}"] == "no_cov", f"{feat}_1"] = "no_cov"
        gb_df.loc[gb_df[f"{feat}"] == "no_cov", f"{feat}_2"] = "no_cov"

    # remove original multi-value feature columns
    gb_df = gb_df.drop(columns=mv_feats)

    return gb_df, gb_df.columns.to_list()[2:]


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def crop(gb_df, perc):
    """Remove languages from dataframe that do not have at least <perc>% feature coverage"""
    tot_feats = len([x for x in gb_df.columns if x.startswith('GB')])
    for i, row in gb_df.iterrows():
        no_data = row.to_list().count('no_cov') + row.to_list().count('?')
        if (tot_feats - no_data) < (perc * tot_feats):
            gb_df = gb_df.drop(i)

    return gb_df


def main():
    args = create_arg_parser()
    mv_feats = ["GB024", "GB025", "GB065", "GB130", "GB193", "GB203"]

    # Load Grambank matrix
    gb_matrix = pd.read_csv(args.gb_file)
    gb_feats = [x for x in gb_matrix.columns.to_list() if x.startswith("GB")]

    # Optional: binarize multi-value features
    if args.binarize:
        gb_matrix, gb_feats = binarize(gb_matrix, mv_feats)
        gb_matrix.to_csv(args.data_output_file)

    # Optional: crop langs according to feature coverage
    if args.crop_perc:  # specify minimum % coverage per lang
        gb_matrix = crop(gb_matrix, args.crop_perc)
        gb_matrix.to_csv(args.data_output_file)

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
        sim_df = sim_df.map(
            normalize, x_min=sim_df.min().min(), x_max=sim_df.max().max()
        )

    sim_df.to_csv(args.output_dist_file)


if __name__ == "__main__":
    main()
