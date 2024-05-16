import argparse
from typdiv.sampling import Sampler, Language, SamplingFunc
from pathlib import Path
import numpy as np
import warnings
import pandas as pd
from typdiv.measures import entropy, fvi, mpd, fvo
import concurrent.futures
from tqdm import tqdm
from itertools import combinations
from dataclasses import dataclass

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
        default=DATA / "gb_vec_sim.csv",  # "gb_vec_sim_0.csv",
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
        default=DATA / "gb_binarized.csv",
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
        default=500,
        help="End of the range of ks to test (inclusive).",
    )
    parser.add_argument(
        "-st",
        type=int,
        default=5,
        help="Step size for the range of ks to test.",
    )
    parser.add_argument(
        "-counts_path",
        type=Path,
        default=DATA / "convenience_counts.json",
        help="File with language counts from previous work.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Result:
    run: int
    ent_score_with: float
    ent_score_without: float
    fvi_score: float
    mpd_score: float
    fvo_score: float


class Evaluator:
    def __init__(self, gb_by_lang: dict[Language, list[str]], distances) -> None:
        self.gb_by_lang = gb_by_lang
        self.n_features = len(gb_by_lang[list(gb_by_lang.keys())[0]])
        self.cache: dict[str, Result] = dict()
        self.distances = distances

    def evaluate_sample(self, sample: list[Language], run: int) -> Result:
        if (sample_key := "".join(sorted(sample))) and sample_key in self.cache:
            return self.cache[sample_key]

        # Language-based methods: MPD, FVO
        pairs = [p for p in combinations(sample, 2)]
        mpd_score = mpd(pairs, self.distances)
        fvo_score = fvo(pairs, self.gb_by_lang)

        # Feature-based methods: Entropy, FVI
        ents_with_missing, ents_without_missing, fvis = [], [], []
        for i in range(self.n_features):
            vals_with_missing, vals_without_missing = [], []
            for lang in sample:
                fv = self.gb_by_lang[lang][i]
                vals_with_missing.append(fv)
                if fv != "?":
                    vals_without_missing.append(fv)
            ents_with_missing.append(entropy("".join(vals_with_missing)))
            ents_without_missing.append(entropy("".join(vals_without_missing)))
            fvis.append(fvi("".join(vals_without_missing)))

        avg_ent_with = sum(ents_with_missing) / len(ents_with_missing)
        avg_ent_without = sum(ents_without_missing) / len(ents_without_missing)
        avg_fvi = sum(fvis) / len(fvis)

        result = Result(
            run, avg_ent_with, avg_ent_without, avg_fvi, mpd_score, fvo_score
        )

        self.cache[sample_key] = result

        return result

    def rand_runs(
        self,
        runs: int,
        func: SamplingFunc,
        N: list[Language],
        k: int,
    ) -> list[Result]:
        results = [
            self.evaluate_sample(func(N, k, run + k), run) for run in range(runs)
        ]
        return results


def main():
    args = create_arg_parser()

    sampler = Sampler(
        dist_path=args.dist_path,
        gb_path=args.gb_path,
        wals_path=args.wals_path,
        counts_path=args.counts_path,
    )

    gb = pd.read_csv(args.gb_features_path, index_col="Lang_ID")
    gb = gb.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

    # no_cov introduces a lot of unneeded entropy and both 'missing' values
    # have the same meaning (roughly) for our purposes
    gb.replace(to_replace="no_cov", value="?", inplace=True)
    gb_by_lang = {i: np.array(row) for i, row in gb.iterrows()}

    dist_df = pd.read_csv(args.dist_path).set_index("Unnamed: 0")
    dist_dict = dist_df.to_dict("dict")  # TODO: this contains double info

    evaluator = Evaluator(gb_by_lang, dist_dict)

    # n runs to get an average for the random methods with a different seed per run
    RUNS = args.rand_runs
    RANGE = range(args.s, args.e + 1, args.st)
    # our frame here is all languages in grambank TODO: get this from args
    N = sorted(gb.index.unique())
    del gb

    # method with n runs
    experiments = [
        # deterministic
        ("mdp", 1),
        ("mmdp", 1),
        ("convenience", 1),  # TODO: this one can become random in the future?
        # random
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
                    }
                )

    pd.DataFrame().from_records(records).to_csv(args.results_path, index=False)


if __name__ == "__main__":
    main()
