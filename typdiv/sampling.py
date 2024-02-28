import pandas as pd
import random
from typdiv.typ_dist import get_summed_dist_dict, get_first_point
from pathlib import Path

Language = str
METHODS = ["random", "random_family", "random_genus", "mdp", "mmdp"]


class Sampler:
    def __init__(self, dist_path: Path, gb_path: Path, wals_path: Path) -> None:
        for p in [dist_path, gb_path, wals_path]:
            if not p.exists():
                raise FileNotFoundError(f"Cannot find {p}")

        self.dist_path = dist_path
        self.gb_path = gb_path
        self.wals_path = wals_path

        self._dist_df = None
        self._gb_df = None
        self._wals_df = None

    def sample_mdp(self, frame: list[Language], k: int, random_seed: int | None = None) -> list[Language]:
        """
        Maximum Diversity Problem
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

        return [id2lang[i] for i in langs]

    def sample_mmdp(self, frame: list[Language], k: int, random_seed: int | None = None) -> list[Language]:
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

        return [id2lang[i] for i in S]

    def sample_random(self, frame: list[Language], k: int, random_seed: int | None = None) -> list[Language]:
        """Sample k from N completely randomly"""
        # local random instance to make this thread safe
        rand = random.Random(random_seed)
        return rand.sample(frame, k)

    def sample_random_family(self, frame: list[Language], k: int, random_seed: int | None = None) -> list[Language]:
        """
        Sample k languages from N where we sample from
        language families as uniformly as the data allows.
        """
        df = self.gb_df[self.gb_df["Glottocode"].isin(frame)]
        return self._df_sample(df, "Family_name", k, random_seed)

    def sample_random_genus(self, frame: list[Language], k: int, random_seed: int | None = None) -> list[Language]:
        """
        Sample k languages from N where we sample from
        language genera as uniformly as the data allows.
        """
        df = self.wals_df[self.wals_df["Glottocode"].isin(frame)]
        return self._df_sample(df, "Genus", k, random_seed)

    def _df_sample(self, df: pd.DataFrame, key: str, k: int, random_seed: int | None = None):
        if k < 1 or k > len(df):
            raise ValueError("Invalid value for k, make sure k > 0 and k <= len(N)")

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
            new_sample = df[~(df["Glottocode"].isin(codes))].sample(1, random_state=random_seed)
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