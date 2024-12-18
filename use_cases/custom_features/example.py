from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from typdiv_sampling.distance import make_language_distances
from typdiv_sampling.evaluation import Evaluator
from typdiv_sampling.sampling import Sampler

TEST_PATH = Path(__file__).parent


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        help="Path to output csv file with results.",
    )

    return parser.parse_args()


def main():
    # An example script to show how to use custom features in all stages of the framework.
    # The steps correspond to the diagram in the readme and the paper.
    args = get_args()

    features_path = TEST_PATH / "example_features.csv"

    # STEP 1: Language Features
    # The file already contains our language features in the format described in the readme.
    df = pd.read_csv(features_path, index_col="Lang_ID")

    # STEP 2: Language Distances
    distances = make_language_distances(df)
    # Save our distances so we don't have to recompute them and use them elsewhere.
    distances_path = TEST_PATH / "example_distances.csv"
    distances.to_csv(distances_path)

    # STEP 3: Sample Languages and evaluate them
    # Initialize the sampler with our new distances, the stratification of languages can stay the same:
    # we still use genus from WALS, family from glottolog/grambank and counts from the typdiv survey.
    sampler = Sampler(dist_path=distances_path)
    evaluator = Evaluator(distances_path=distances_path, features_path=features_path)

    # Create some samples
    frame = distances.index.tolist()  # our frame is all languages in the distance file
    methods = ["maxsum", "maxmin", "convenience", "random", "random_family", "random_genus"]
    k_languages = 5

    records = []
    for method in methods:
        sample = getattr(sampler, f"sample_{method}")(frame, k_languages)
        result = evaluator.evaluate_sample(sample)
        records.append(
            {
                "method": method,
                "run": result.run,
                "entropy_with_missing": result.ent_score_with,
                "entropy_without_missing": result.ent_score_without,
                "fvi": result.fvi_score,
                "mpd": result.mpd_score,
                "fvo": result.fvo_score,
                "k": k_languages,
                "sample": ",".join(sorted(result.sample)),
            }
        )

    pd.DataFrame().from_records(records).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
