import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from typdiv_sampling.constants import (
    DEFAULT_GB_FEATURES_PATH,
    DEFAULT_GB_RAW_FEATURES_PATH,
    DEFAULT_LANGUOID_PATH,
    GB_MULTI_VALUE_FEATURES,
    DEFAULT_DISTANCES_PATH,
)
from typdiv_sampling.distance import make_language_distances


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gb_path",
        type=Path,
        default=DEFAULT_GB_RAW_FEATURES_PATH,
        help="Path to Grambank file with features per language.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=DEFAULT_GB_FEATURES_PATH,
        help="Name of file that post-processed (bin/norm/crop/filter) grambank version should be written to",
    )
    parser.add_argument(
        "-d",
        "--distances_path",
        type=Path,
        default=DEFAULT_DISTANCES_PATH,
        help="Path to output file with langauge distances.",
    )
    parser.add_argument(
        "-b",
        "--binarize",
        action="store_true",
        help="Option to binarize multi-value features",
    )
    parser.add_argument(
        "-c",
        "--crop",
        type=float,
        help="Option: specify minimum feature coverage percentage per language",
    )
    parser.add_argument(
        "-r",
        "--remove_macro",
        action="store_true",
        help="Option: remove macro-languages",
    )
    parser.add_argument(
        "-la",
        "--languoids_path",
        type=Path,
        default=DEFAULT_LANGUOID_PATH,
        help="Path to Glottolog file with languoid metadata (for filtering macro-languages).",
    )
    parser.add_argument(
        "-i",
        "--include_features",
        nargs="+",
        type=str,
        help="List of feature names (columns) that should be included, the rest is ignored.",
    )
    parser.add_argument(
        "-l",
        "--include_languages",
        nargs="+",
        type=str,
        help="List of glottocodes (rows) that should be included, the rest is ignored.",
    )
    parser.add_argument(
        "-n",
        "--normalize_dists",
        action="store_true",
        help="Normalize language distances.",
    )
    args = parser.parse_args()
    return args


def binarize(gb_df, selected_mv_feats: list[str]):
    """Binarize multi-value features"""

    for feat in selected_mv_feats:
        # label 3 (both)
        gb_df[f"{feat}_1"] = (gb_df[feat] == 1).astype(float)
        gb_df[f"{feat}_2"] = (gb_df[feat] == 2).astype(float)

        # label 3 (both)
        gb_df.loc[gb_df[feat] == 3, f"{feat}_1"] = 1.0
        gb_df.loc[gb_df[feat] == 3, f"{feat}_2"] = 1.0

        # label 0 (none)
        gb_df.loc[gb_df[feat] == 0, f"{feat}_1"] = 0.0
        gb_df.loc[gb_df[feat] == 0, f"{feat}_2"] = 0.0

        # if original value was nan ('?' or 'no_cov'), put this back
        gb_df.loc[gb_df[feat].isna(), f"{feat}_1"] = float("nan")
        gb_df.loc[gb_df[feat].isna(), f"{feat}_2"] = float("nan")

    # remove original multi-value feature columns
    gb_df = gb_df.drop(columns=selected_mv_feats)

    return gb_df


def main():
    args = create_arg_parser()

    # Load Grambank matrix
    df = pd.read_csv(args.gb_path, index_col="Lang_ID")
    # Make sure we have only numbers
    df = df.replace(["no_cov", "?"], np.nan).astype(float)

    # Optional: include only a subset of features
    if incl_feats := args.include_features:
        before = df.shape
        df = df[incl_feats]
        after = df.shape
        print(f"Incl features {before=} and {after=}")

    # Optional: include only a subset of languages
    if incl_langs := args.include_languages:
        before = df.shape
        df = df[df.index.isin(incl_langs)]
        after = df.shape
        print(f"Incl features {before=} and {after=}")

    # Optional: binarize multi-value features
    if args.binarize:
        before = df.shape
        # We have possibly removed values in the first step
        to_process = sorted(list(set(GB_MULTI_VALUE_FEATURES).intersection(set(df.columns))))
        df = binarize(df, to_process)
        after = df.shape
        print(f"Binarize {before=} and {after=}")

    # Optional: remove macrolanguages
    if args.remove_macro:
        if not (languoids_path := args.languoids_path):
            raise FileNotFoundError("To remove macro languages, please provide the 'languoids_path'.")
        before = df.shape
        glottolog_data = pd.read_csv(languoids_path)
        df = df[df.index.isin(glottolog_data[glottolog_data["child_language_count"] == 0]["id"])]
        after = df.shape
        print(f"Macro {before=} and {after=}")

    # Optional: crop langs according to feature coverage
    if crop_percentage := args.crop:  # specify minimum % coverage per lang
        before = df.shape
        threshold = crop_percentage * df.shape[1]
        print(f"{threshold=}")
        print(df.notnull().sum(axis=1).to_string())
        df = df[df.notnull().sum(axis=1) > threshold]
        after = df.shape
        print(f"Crop {before=} and {after=}")

    # Save (processed) Grambank version
    df.astype("Int64", errors="ignore").to_csv(args.output_path, index="Lang_ID")

    # At this point we have the data in the correct format to create distances
    distances = make_language_distances(df, args.normalize_dists)
    distances.to_csv(args.distances_path)


if __name__ == "__main__":
    main()
