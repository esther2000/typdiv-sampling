import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from typdiv_sampling.evaluation import Evaluator, Result
from typdiv_sampling.sampling import Sampler
from typdiv_sampling.constants import (
    DEFAULT_DISTANCES_PATH,
    DEFAULT_GB_PATH,
    DEFAULT_WALS_PATH,
    DEFAULT_GB_FEATURES_PATH,
    DEFAULT_COUNTS_PATH,
)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dist_path",
        type=Path,
        default=DEFAULT_DISTANCES_PATH,
        help="File with pairwise language distances.",
    )
    parser.add_argument(
        "-gb_path",
        type=Path,
        default=DEFAULT_GB_PATH,
        help="File with Grambank language information.",
    )
    parser.add_argument(
        "-wals_path",
        type=Path,
        default=DEFAULT_WALS_PATH,
        help="File with WALS language information.",
    )
    parser.add_argument(
        "-gb_features_path",
        type=Path,
        default=DEFAULT_GB_FEATURES_PATH,
        help="File with Grambank features.",
    )
    parser.add_argument(
        "-counts_path",
        type=Path,
        default=DEFAULT_COUNTS_PATH,
        help="File with language counts from previous work.",
    )
    parser.add_argument(
        "-r",
        "--results_path",
        type=Path,
        required=True,
        help="CSV to write results to.",
    )
    parser.add_argument(
        "-f",
        "--frame_path",
        type=Path,
        help="Frame to sample from as a text file with one Glottocode per line. If left empty, the sampling frame "
        "will be all languages in the specified language distance file.",
    )
    parser.add_argument(
        "-n_cpu",
        type=int,
        default=8,
        help="Number of cpu cores to use.",
    )
    parser.add_argument(
        "-rand_runs",
        type=int,
        default=10,
        help="Number of runs per k for random methods.",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=5,
        help="Start of the range of ks to test.",
    )
    parser.add_argument(
        "-e",
        type=int,
        default=140,
        help="End of the range of ks to test (inclusive).",
    )
    parser.add_argument(
        "-st",
        type=int,
        default=5,
        help="Step size for the range of ks to test.",
    )
    return parser.parse_args()


def main():
    args = create_arg_parser()

    sampler = Sampler(
        dist_path=args.dist_path,
        gb_path=args.gb_path,
        wals_path=args.wals_path,
        counts_path=args.counts_path,
    )

    # TODO: remove this pre-processing
    dist_df = pd.read_csv(args.dist_path).set_index("Unnamed: 0")
    evaluator = Evaluator(args.gb_features_path, args.dist_path)

    # n runs to get an average for the random methods with a different seed per run
    RUNS = args.rand_runs
    RANGE = [args.s] if args.s == args.e else range(args.s, args.e + 1, args.st)
    if args.frame_path:
        N = sorted(
            [g_code.strip() for g_code in args.frame_path.read_text().split("\n")]
        )
    else:
        N = sorted(dist_df.columns.to_list())

    # method with n runs
    experiments = [
        # deterministic
        ("maxsum", 1),
        ("maxmin", 1),
        # random
        ("convenience", RUNS),
        ("random", RUNS),
        ("random_family", RUNS),
        ("random_genus", RUNS),
    ]

    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_cpu) as ex:
        futures = {}
        for k in RANGE:
            for name, runs in experiments:
                future = ex.submit(
                    evaluator.rand_runs, runs, getattr(sampler, f"sample_{name}"), N, k
                )
                # some trickery, we use the object hash of a future as a dict key
                futures[future] = (name, k)

        for res in tqdm(
            concurrent.futures.as_completed(futures.keys()),
            desc="Processing",
            total=len(futures),
        ):
            method, k = futures[res]
            for run_res in res.result():
                run_res: Result
                records.append(
                    {
                        "method": method,
                        "run": run_res.run,
                        "entropy_with_missing": run_res.ent_score_with,
                        "entropy_without_missing": run_res.ent_score_without,
                        "fvi": run_res.fvi_score,
                        "mpd": run_res.mpd_score,
                        "fvo": run_res.fvo_score,
                        "k": k,
                        "sample": ",".join(sorted(run_res.sample)),
                    }
                )

    pd.DataFrame().from_records(records).to_csv(args.results_path, index=False)


if __name__ == "__main__":
    main()
