#!/usr/bin/python3
"""
Description:    Calculate distances between pairs of languages based on Grambank feature values
Usage:          python compute_all_distances.py -g <GRAMBANK_FOLDER> -o <OUTPUT_FILE> -b (optional)
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances

from typdiv_sampling.constants import (
    DEFAULT_DIST_PATH,
    DEFAULT_GB_FEATURES_PATH,
    DEFAULT_GB_RAW_FEATURES_PATH,
    DEFAULT_LANGUOID_PATH,
)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dist_file",
        type=Path,
        default=DEFAULT_DIST_PATH,
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
        default=DEFAULT_GB_FEATURES_PATH,
        help="Name of file that post-processed (bin/norm/crop/filter) grambank version should be written to",
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
        "-r",
        "--remove_macro",
        action="store_true",
        help="Option: remove macrolanguages",
    )
    parser.add_argument(
        "-la",
        "--languoids_file",
        default=DEFAULT_LANGUOID_PATH,
        help="Path to Glottolog file with languoid metadata.",
    )
    parser.add_argument(
        "-g",
        "--gb_file",
        default=DEFAULT_GB_RAW_FEATURES_PATH,
        help="Path to Grambank file with features per language.",
    )
    parser.add_argument(
        "-f",
        "--features",
        help="Path to file with the GB feature IDs (one per line) that should be included."
        "See example: data/feature_subset_example.txt",
    )
    parser.add_argument(
        "-l",
        "--select_langs",
        help="Path to file with the language glottocodes (one per line) that should be included.",
    )

    args = parser.parse_args()
    return args


def binarize(gb_df, selected_mv_feats):
    """Binarize multi-value features
    TODO: this could probably be done more elegantly but it works"""
    for feat in selected_mv_feats:
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
    gb_df = gb_df.drop(columns=selected_mv_feats)

    return gb_df, gb_df.columns.to_list()[2:]


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def crop(gb_df, perc):
    """Remove languages from dataframe that do not have at least <perc>% feature coverage"""
    tot_feats = len([x for x in gb_df.columns if x.startswith("GB")])
    for i, row in gb_df.iterrows():
        no_data = row.to_list().count("no_cov") + row.to_list().count("?")
        if (tot_feats - no_data) < (perc * tot_feats):
            gb_df = gb_df.drop(i)

    return gb_df


def filter_macrolangs(gb_df, languoid_path):
    """Filter out macrolanguages (e.g. Central pacific linkage)"""
    glottolog_data = pd.read_csv(languoid_path)
    child_langs = {
        row["id"]: row["child_language_count"] for _, row in glottolog_data.iterrows()
    }

    for i, row in gb_df.iterrows():
        if child_langs[row["Lang_ID"]] > 0:
            gb_df = gb_df.drop(i)

    return gb_df


def filter_langs(gb_df, langs):
    """Filter out languages from list"""
    for i, row in gb_df.iterrows():
        if row["Lang_ID"] not in langs:
            gb_df = gb_df.drop(i)

    return gb_df


def main():
    args = create_arg_parser()
    mv_feats = ["GB024", "GB025", "GB065", "GB130", "GB193", "GB203"]

    # Load Grambank matrix
    gb_matrix = pd.read_csv(args.gb_file)
    gb_feats = [x for x in gb_matrix.columns.to_list() if x.startswith("GB")]

    # Optional: include only a subset of features
    if args.features:
        with open(args.features) as feat_file:
            incl_feats = [x.rstrip() for x in feat_file.readlines()]
        gb_matrix = gb_matrix[["Unnamed: 0", "Lang_ID"] + incl_feats]
        gb_feats = incl_feats

    # Optional: include only a subset of languages
    if args.select_langs:
        with open(args.select_langs) as lang_file:
            incl_langs = [x.rstrip() for x in lang_file.readlines()]
        gb_matrix = filter_langs(gb_matrix, incl_langs)

    # Optional: binarize multi-value features
    if args.binarize:
        gb_matrix, gb_feats = binarize(gb_matrix, set(mv_feats).intersection(gb_feats))

    # Optional: remove macrolanguages
    if args.remove_macro:
        gb_matrix = filter_macrolangs(gb_matrix, args.languoids_file)

    # Optional: crop langs according to feature coverage
    if args.crop_perc:  # specify minimum % coverage per lang
        gb_matrix = crop(gb_matrix, args.crop_perc)

    # Save (processed) Grambank version
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
    sim_df = pd.DataFrame(sim_matrix, columns=langs, index=langs).fillna(
        0
    )  # NOTE: 0 for maximisation only!
    if args.normalize:
        sim_df = sim_df.map(
            normalize, x_min=sim_df.min().min(), x_max=sim_df.max().max()
        )

    sim_df.to_csv(args.output_dist_file)

    # with open('gb_frame-bnrc75.txt', 'w') as frame:
    #  for l in langs:
    # frame.write(l+"\n")


if __name__ == "__main__":
    main()
