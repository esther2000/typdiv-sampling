"""
Description:    typological diversity evaluation
Usage:
NOTE: This is a work in progress that only prints entropy for now
"""

import argparse
import pandas as pd
import math
from pathlib import Path

SCRIPT_WD = Path(__file__).resolve().parent


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sample",
        type=str,
        default=SCRIPT_WD / "samples/random_genus-langs_gb-10-gb_vec_sim_0-123.txt",
        help="File with languages to select from, one language code per line.",
    )
    parser.add_argument(
        "-gb",
        "--grambank",
        type=str,
        default=SCRIPT_WD.parent / "data/gb_lang_feat_vals.csv",
        help="Path to Grambank feature csv.",
    )
    return parser.parse_args()


def entropy(string):
    """Calculates the Shannon entropy of a string
    from: https://stackoverflow.com/questions/67059620/calculate-entropy-from-binary-bit-string
    """

    # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in set(list(string))]

    # calculate the entropy
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])

    return entropy


def main():

    args = create_arg_parser()

    with open(args.sample) as langsfile:
        langs = [x.strip() for x in langsfile.readlines()]

    gb_matrix = pd.read_csv(args.grambank)

    df = gb_matrix.loc[gb_matrix["Lang_ID"].isin(langs)]
    feat_vecs = {name: values.to_list() for name, values in df.items()}

    entropies = []
    for feat, vec in feat_vecs.items():
        if feat.startswith("GB"):
            filtered_vec = [x for x in vec if x == "0" or x == "1"]
            binary_string = "".join(filtered_vec)
            e = entropy(binary_string)
            entropies.append(e)

    print(f"Entropy: {sum(entropies) / len(entropies)}")


if __name__ == "__main__":
    main()
