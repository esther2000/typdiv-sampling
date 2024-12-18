import argparse
import itertools

import geopy.distance
import numpy as np
import pandas as pd
from typdiv_sampling.constants import DEFAULT_GB_LANGUAGES_PATH
from tqdm import tqdm


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default="geo_dists.csv",
        help="Filename to write coordinate distances (km) to.",
    )
    return parser.parse_args()


def dict_to_matrix(dist_dict, langs):
    """This function is adapted from get_dist_matrix() in typdiv_sampling.matrix
    TODO: make the try-except stuff more elegant (sort l1, l2?)"""
    n, out = len(langs), []
    for i in range(n):
        dists = []
        for j in range(n):
            try:
                dists.append(dist_dict[(langs[i], langs[j])])
            except KeyError:
                try:
                    dists.append(dist_dict[(langs[j], langs[i])])
                except KeyError:
                    dists.append(0)  # for maximisation: small value
        out.append(dists)

    return np.array(out)


def main():
    args = create_arg_parser()

    # retrieve language coordinates from grambank
    gb_df = pd.read_csv(DEFAULT_GB_LANGUAGES_PATH)
    coords = {row["ID"]: row[["Latitude", "Longitude"]].to_list() for _, row in gb_df.iterrows()}

    # calculate pairwise km distances (may take a few minutes)
    km_dists = dict()
    pairs = list(itertools.combinations(coords.keys(), 2))
    for l1, l2 in tqdm(pairs, desc="Calculating pairs"):
        try:
            km_dists[(l1, l2)] = geopy.distance.geodesic(coords[l1], coords[l2]).km
        except ValueError:
            pass  # no coordinates in database

    # convert to matrix
    gb_ids = sorted(gb_df["ID"].to_list())
    mtx = dict_to_matrix(km_dists, gb_ids)

    # write to csv
    matrix_df = pd.DataFrame(mtx, columns=gb_ids, index=gb_ids).fillna(0)
    matrix_df.to_csv(args.outfile)


if __name__ == "__main__":
    main()
