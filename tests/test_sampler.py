from pathlib import Path
from typdiv.sampling import Sampler
import pytest

CWD = Path(__file__).parent
FIXTURES = CWD / "fixtures"


def test_df_sampling():
    sampler = Sampler(
        dist_path=FIXTURES / "fake_dists.csv",
        gb_path=FIXTURES / "fake_gb.csv",
        wals_path=FIXTURES / "fake_wals.csv",
    )

    """
    GB test subset:
        100 languages
        36 families (one with missing value, so 35 usable ones)
    WALS test subset:
        58 languages
        46 genera
    Distances test subset:
        covers distances between all 100 languages
    """

    gb_frame = [l.strip() for l in (FIXTURES / "gb_frame_100.txt").read_text().split("\n")]

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

    with pytest.raises(ValueError, match="Invalid value for k"):
        sampler.sample_random_genus(gb_frame, 100)

    with pytest.raises(ValueError, match="Invalid value for k"):
        sampler.sample_random_genus(gb_frame, 0)