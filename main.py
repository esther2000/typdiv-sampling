"""
Description:    Sample k languages from N, using different methods
Usage:          python main.py -s <SAMPLING_METHODS> -k <NUM_LANGS> -f <ALL_LANGS> -d <LANG_SIM>
"""

import argparse
from typdiv.sampling import Sampler, METHODS
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# location of this file, so it does not matter from where this script is called
CWD = Path(__file__).parent
DATA = CWD / "data"


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sampling_method",
        nargs="+",
        type=str,
        default="typ_mdp",
        help=f"Sampling method(s), choose from: {','.join(METHODS)}.",
    )
    parser.add_argument(
        "-k",
        "--k_langs",
        type=int,
        default=10,
        help="Number of languages to select from selection.",
    )
    parser.add_argument(
        "-f",
        "--frame_path",
        type=Path,
        default=DATA / "frames/langs_gb.txt",
        help="File with languages to select from, one Glottocode per line.",
    )
    parser.add_argument(
        "-d",
        "--dist_path",
        type=Path,
        default=DATA / "gb_vec_sim_0.csv",
        help="File with pairwise language distances.",
    )
    parser.add_argument(
        "-gb_path",
        type=Path,
        default=CWD / "grambank/cldf/languages.csv",
        help="File with Grambank language information.",
    )
    parser.add_argument(
        "-wals_path",
        type=Path,
        default=DATA / "wals_dedup.csv",
        help="File with WALS language information.",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=123,
        help="Random seed.",
    )

    return parser.parse_args()


def main():

    args = create_arg_parser()

    # define sampling frame
    with open(args.frame_path) as all_langs:
        frame = [x.strip() for x in all_langs.readlines()]

    sampler = Sampler(
        dist_path=args.dist_path,
        gb_path=args.gb_path,
        wals_path=args.wals_path,
    )

    for method in args.sampling_method:
        if method not in METHODS:
            print(f"Skipping unknown method {method}")
            continue

        sample = getattr(sampler, f"sample_{method}")(frame, args.k_langs, args.seed)
        outfile = f"evaluation/samples/{method}-{args.frame_path.stem}-{args.k_langs}-{args.dist_path.stem}-{args.seed}.txt"
        Path(outfile).write_text("\n".join(sample))


if __name__ == "__main__":
    main()
