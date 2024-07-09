import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

from typdiv_sampling.distance import get_first_point, get_summed_dist_dict

Language = str
SamplingFunc = Callable[[list[Language], int, int], list[Language]]
METHODS = ["random", "random_family", "random_genus", "maxsum", "maxmin", "convenience"]


class Sampler:
    def __init__(
        self, dist_path: Path, gb_path: Path, wals_path: Path, counts_path: Path
    ) -> None:
        for p in [dist_path, gb_path, wals_path, counts_path]:
            if not p.exists():
                raise FileNotFoundError(f"Cannot find {p}")

        self.dist_path = dist_path
        self.gb_path = gb_path
        self.wals_path = wals_path
        self.counts_path = counts_path

        self._dist_df = None
        self._gb_df = None
        self._wals_df = None
        self._counts = None

    def sample_maxsum(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """
        Maximum Diversity Problem (MaxSum)
        Sample k languages from N, where we iteratively add the
        next point that yields the largest summed distance (greedy).
        """
        dists, id2lang = self.get_dists(frame)

        most_distant_langid = dists.sum(axis=1).argmax()
        all_langs = [i for i in range(len(dists))]

        langs = [most_distant_langid]
        all_langs.remove(most_distant_langid)
        while len(langs) <= k - 1:
            summed_dist = get_summed_dist_dict(dists, all_langs, langs)
            next_most_distant = max(summed_dist, key=lambda x: summed_dist[x])
            all_langs.remove(next_most_distant)
            langs.append(next_most_distant)

        return sorted([id2lang[i] for i in langs])

    def sample_maxmin(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """
        MaxMin Diversity Problem
        Sample k languages from N, where we iteratively add the
        next point that yields the maximum minimum distance between
        any two points in k.
        """
        dists, id2lang = self.get_dists(frame)

        p1 = get_first_point(dists)
        p2 = dists[p1].argmax()

        L = {i for i in range(dists.shape[0])}
        S = {p1, p2}

        while len(S) < k:
            rest_L = tuple(L.symmetric_difference(S))
            rest_dists = dists[rest_L, :].T[tuple(S), :].T
            S.add(rest_L[rest_dists.min(axis=1).argmax()])

        return sorted([id2lang[i] for i in S])

    def sample_random(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """Sample k from N completely randomly"""
        # local random instance to make this thread safe
        rand = random.Random(random_seed)
        return sorted(rand.sample(frame, k))

    def sample_random_family(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """
        Sample k languages from N where we sample from
        language families as uniformly as the data allows.
        """
        df = self.gb_df[self.gb_df["Glottocode"].isin(frame)]
        return sorted(self._df_sample(df, "Family_name", k, random_seed))

    def sample_random_genus(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """
        Sample k languages from N where we sample from
        language genera as uniformly as the data allows.
        """
        df = self.wals_df[self.wals_df["Glottocode"].isin(frame)]
        return sorted(self._df_sample(df, "Genus", k, random_seed))

    def sample_convenience(
        self, frame: list[Language], k: int, random_seed: int | None = None
    ) -> list[Language]:
        """
        Sample k most-used languages from the literature on 'typologically diverse' language samples
        TODO: this can be max 195 --> change experiments to less than 500? or just lower number for this baseline?
        """
        rand = random.Random(random_seed)

        # Filter so we're using only those language that are in our frame
        grouped_counts = defaultdict(list)
        n_langs = 0
        for lang, lang_count in self.counts:
            if lang not in frame:
                continue
            n_langs += 1
            # Group by count and shuffle lists so ties are random
            grouped_counts[lang_count].append(lang)

        if n_langs < k:
            raise ValueError(
                f"Invalid value {k=}, we only have {n_langs} languages to sample from."
            )

        # Flatten list
        counts = []
        for count, languages in grouped_counts.items():
            shuffled_langs = rand.sample(languages, k=len(languages))
            for lang in shuffled_langs:
                counts.append((lang, count))

        return sorted([lang for lang, _ in counts[:k]])

    def _df_sample(
        self, df: pd.DataFrame, key: str, k: int, random_seed: int | None = None
    ):
        if k < 1 or k > len(df):
            raise ValueError(f"Invalid value {k=}, make sure k > 0 and k <= {len(df)}")

        codes = (
            # Sample one language from each key
            df.groupby(key)
            .apply(lambda x: x.sample(1, random_state=random_seed))
            .reset_index(drop=True)["Glottocode"]
            .tolist()
        )

        # We're lucky, n groups is the same as the number of langs we're looking for
        if len(codes) == k:
            return codes

        # We have more languages than we need, sample what we need
        if len(codes) > k:
            rand = random.Random(random_seed)
            return rand.sample(codes, k)

        # Figure out how many to get from the groups (as evenly as possible)
        n_groups = len(df[key].unique())
        s_size = k // n_groups
        codes = (
            df.groupby(key)
            # the min here makes sure we also sample from groups that have
            # fewer members than the requested sample size
            .apply(lambda x: x.sample(min(len(x), s_size), random_state=random_seed))
            .reset_index(drop=True)["Glottocode"]
            .tolist()
        )
        # Fill up the remaining langs by random selection from all groups
        while len(codes) < k:
            new_sample = df[~(df["Glottocode"].isin(codes))].sample(
                1, random_state=random_seed
            )
            codes.append(new_sample["Glottocode"].values[0])

        return codes

    def get_dists(self, frame: list[Language]):
        """Get language distances for the given frame."""
        dists = self.dist_df[frame].loc[frame]
        id2lang = dists.columns.tolist()
        dists = dists.to_numpy()
        return dists, id2lang

    @property
    def dist_df(self):
        """Language distances dataframe, lazily loaded."""
        if self._dist_df is None:
            self._dist_df = pd.read_csv(self.dist_path, index_col=0)
        return self._dist_df

    @property
    def gb_df(self):
        """Grambank languages dataframe, lazily loaded."""
        if self._gb_df is None:
            self._gb_df = pd.read_csv(self.gb_path)
        return self._gb_df

    @property
    def wals_df(self):
        """WALS languages dataframe, lazily loaded."""
        if self._wals_df is None:
            self._wals_df = pd.read_csv(self.wals_path)
        return self._wals_df

    @property
    def counts(self):
        """Language counts list, lazily loaded."""
        if self._counts is None:
            with open(self.counts_path, "r") as s:
                counts = json.load(s)
            self._counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return self._counts
