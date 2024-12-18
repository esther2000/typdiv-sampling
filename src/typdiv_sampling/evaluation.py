import itertools
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


from typdiv_sampling.sampling import Language, SamplingFunc
from typdiv_sampling.constants import DEFAULT_GB_FEATURES_PATH, DEFAULT_DISTANCES_PATH


def entropy(string: str) -> float:
    """
    Calculates the Shannon entropy of a string
    from: https://stackoverflow.com/questions/67059620/calculate-entropy-from-binary-bit-string
    """
    # get probability of chars in string
    prob = [string.count(c) / len(string) for c in set(list(string))]
    entropy = -sum(p * math.log(p) / math.log(2.0) for p in prob)

    return entropy


def fvi(string: str) -> float:
    """
    Return the number of unique feature values included for a binary feature string.
    """
    return len(set(string)) / 2


def mpd(pairs: list, distances: dict) -> float:
    """
    Calculate mean pairwise distance for each string
    """
    mpds = []
    for pair in pairs:
        mpds.append(distances[pair[0]][pair[1]])

    return sum(mpds) / len(mpds)


def fvo(pairs: list, feats: dict) -> float:
    """
    Calculate fraction of feature value overlap per language pair in sample
    """
    fracs = []
    for pair in pairs:
        same, total = 0, 0
        l1_feats, l2_feats = feats[pair[0]], feats[pair[1]]
        for f1, f2 in zip(l1_feats, l2_feats):
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 == f2:
                same += 1
            total += 1
            try:
                fracs.append(same / total)
            except ZeroDivisionError:
                pass  # TODO: or append 0?

    return sum(fracs) / len(fracs)


@dataclass(frozen=True)
class Result:
    run: int | None
    ent_score_with: float
    ent_score_without: float
    fvi_score: float
    mpd_score: float
    fvo_score: float
    sample: set[Language]


class Evaluator:
    """Class to evaluate language samples."""

    def __init__(
        self,
        features_path: Path = DEFAULT_GB_FEATURES_PATH,
        distances_path: Path = DEFAULT_DISTANCES_PATH,
    ) -> None:
        features = pd.read_csv(features_path, index_col="Lang_ID")
        dist_df = pd.read_csv(distances_path, index_col="Lang_ID")

        dist_dict = dist_df.to_dict("dict")  # TODO: this contains double info
        features_by_lang = {i: np.array(row) for i, row in features.iterrows()}

        self.features_by_lang = features_by_lang
        self.n_features = features.shape[1]
        self.cache: dict[str, Result] = dict()
        self.distances = dist_dict

    def evaluate_sample(self, sample: list[Language], run: int | None = None) -> Result:
        if (sample_key := "".join(sorted(sample))) and sample_key in self.cache:
            return self.cache[sample_key]

        # Language-based methods: MPD, FVO
        pairs = list(itertools.combinations(sample, 2))
        mpd_score = mpd(pairs, self.distances)
        fvo_score = fvo(pairs, self.features_by_lang)

        # Feature-based methods: Entropy, FVI
        ents_with_missing, ents_without_missing, fvis = [], [], []
        for i in range(self.n_features):
            vals_with_missing, vals_without_missing = [], []
            for lang in sample:
                fv = self.features_by_lang[lang][i]
                vals_with_missing.append(fv)
                if not np.isnan(fv):
                    vals_without_missing.append(fv)

            feature_str_with_missing = "".join(str(int(f)) if not np.isnan(f) else "?" for f in vals_with_missing)
            feature_str = "".join(str(int(f)) for f in vals_without_missing if not np.isnan(f))

            ents_with_missing.append(entropy(feature_str_with_missing))
            ents_without_missing.append(entropy(feature_str))
            fvis.append(fvi(feature_str))

        avg_ent_with = sum(ents_with_missing) / len(ents_with_missing)
        avg_ent_without = sum(ents_without_missing) / len(ents_without_missing)
        avg_fvi = sum(fvis) / len(fvis)

        result = Result(
            run,
            avg_ent_with,
            avg_ent_without,
            avg_fvi,
            mpd_score,
            fvo_score,
            set(sample),
        )

        self.cache[sample_key] = result

        return result

    def rand_runs(self, runs: int, func: SamplingFunc, N: list[Language], k: int) -> list[Result]:
        """Runs the provided function 'runs' times with different random seeds."""
        results = [self.evaluate_sample(func(N, k, run + k), run) for run in range(runs)]
        return results
