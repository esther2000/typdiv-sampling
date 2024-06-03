from pathlib import Path

import pytest
from typdiv_sampling import Sampler

CWD = Path(__file__).parent
FIXTURES = CWD / "fixtures"


def test_df_sampling():
    sampler = Sampler(
        dist_path=FIXTURES / "fake_dists.csv",
        gb_path=FIXTURES / "fake_gb.csv",
        wals_path=FIXTURES / "fake_wals.csv",
        counts_path=FIXTURES / "fake_counts.json",
    )

    """
    GB test subset:
        100 languages
        36 families (one with missing value, so 35 usable ones)
    WALS test subset:
        58 languages
        46 genera
    Convenience:
        100 GB languages with fake counts
    Distances test subset:
        covers distances between all 100 languages (a lot of ties)
    """

    gb_frame = [
        lang.strip() for lang in (FIXTURES / "gb_frame_100.txt").read_text().split("\n")
    ]

    # sample what we need, randomly from the families
    assert len(sampler.sample_random_family(gb_frame, 34)) == 34
    # k == len(N), note that this only considers the 35 usable ones, maybe change this later
    # pandas groupby drops nans by default, that's what we do now
    assert len(sampler.sample_random_family(gb_frame, 35)) == 35
    # sample multiple evenly from families
    assert len(sampler.sample_random_family(gb_frame, 80)) == 80

    # same deal, but with wals genera instead
    assert len(sampler.sample_random_genus(gb_frame, 34)) == 34
    assert len(sampler.sample_random_genus(gb_frame, 46)) == 46
    assert len(sampler.sample_random_genus(gb_frame, 55)) == 55

    assert len(sampler.sample_convenience(gb_frame, 80)) == 80

    conv_a = sampler.sample_convenience(gb_frame, 80, random_seed=1)
    conv_b = sampler.sample_convenience(gb_frame, 80, random_seed=2)

    assert conv_a != conv_b

    with pytest.raises(ValueError, match="Invalid value"):
        sampler.sample_convenience(gb_frame, 101)

    with pytest.raises(ValueError, match="Invalid value"):
        sampler.sample_random_genus(gb_frame, 100)

    with pytest.raises(ValueError, match="Invalid value"):
        sampler.sample_random_genus(gb_frame, 0)
