import argparse
from typdiv.sampling import Sampler, METHODS
from pathlib import Path
import numpy as np
import warnings
import pandas as pd
from typdiv.measures import entropy
import concurrent.futures
from tqdm import tqdm


warnings.filterwarnings("ignore", category=DeprecationWarning)

# location of this file, so it does not matter from where this script is called
CWD = Path(__file__).parent
DATA = CWD / "data"


def create_arg_parser():
    parser = argparse.ArgumentParser()
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
        "-gb_features_path",
        type=Path,
        default=DATA / "gb_lang_feat_vals.csv",
        help="File with Grambank features.",
    )
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        required=True,
        help="CSV to write results to.",
    )
    parser.add_argument(
        "-n_cpu",
        type=int,
        default=8,
        help="Number of cpu cores to use.",
    )

    return parser.parse_args()


def evaluate_sample(sample):
    ents = [entropy_by_lang_filtered[lang] for lang in sample]
    return sum(ents) / len(ents)


def rand_runs(runs, func, N, k):
    scores = [evaluate_sample(func(N, k, run + k)) for run in range(runs)]
    return sum(scores) / runs


def main():
    args = create_arg_parser()

    sampler = Sampler(
        dist_path=args.dist_path,
        gb_path=args.gb_path,
        wals_path=args.wals_path,
    )

    gb = pd.read_csv(args.gb_features_path, index_col="Lang_ID")
    gb = gb.drop(["Unnamed: 0"], axis=1)

    gb_by_lang = {lang_id: np.array(row) for lang_id, row in gb.iterrows()}

    # pre compute entropy for all languages since it will not change
    global entropy_by_lang_filtered  # yeah yeah I know
    entropy_by_lang_filtered = {
        # TODO: what to do about missing values
        # here we filter out ? and no_cov to calc entropy
        k: entropy("".join(str(int(x)) for x in v if x not in ["?", "no_cov"]))
        for k, v in gb_by_lang.items()
    }

    # TODO: put this in args and allow for other frames, distances and evaluation metrics
    RUNS = 10  # n runs to get an average for the random methods with a different seed per run
    RANGE = range(5, 505, 5)
    N = sorted(gb_by_lang.keys())  # our frame here is all languages in grambank
    records = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_cpu) as ex:
        # some trickery, we use the object hash of a future as a dict key
        futures = {}
        for k in RANGE:
            # these are deterministic, so no need for runs or seeds
            futures[ex.submit(sampler.sample_mdp, N, k)] = ("mdp", k)
            futures[ex.submit(sampler.sample_mmdp, N, k)] = ("mmdp", k)

            futures[ex.submit(rand_runs, RUNS, sampler.sample_random, N, k)] = ("random", k)
            futures[ex.submit(rand_runs, RUNS, sampler.sample_random_family, N, k)] = ("random_family", k)
            futures[ex.submit(rand_runs, RUNS, sampler.sample_random_genus, N, k)] = ("random_genus", k)

        for res in tqdm(concurrent.futures.as_completed(futures.keys()), desc="Processing", total=len(futures)):
            method, k = futures[res]
            # here we have the samples, not yet evaluated
            if method in ["mdp", "mmdp"]:
                records.append({"method": method, "entropy": evaluate_sample(res.result()), "k": k})
            # and here the average of the random runs
            else:
                records.append({"method": method, "entropy": res.result(), "k": k})

    pd.DataFrame().from_records(records).to_csv(args.results_path, index=False)


if __name__ == "__main__":
    main()
