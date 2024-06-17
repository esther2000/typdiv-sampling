"""
Note: this code is copied from plot.ipynb to provide a non-notebook version of the code.
"""

import pandas as pd
import altair as alt
from pathlib import Path
import argparse

CWD = Path(__file__).parent


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results_file",
        type=Path,
        help="File with results for sampling method, metrics and sample size.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        help="Path and filename for output graph.",
    )
    return parser.parse_args()


def main():
    args = create_arg_parser()

    # Frame = all languages in GB
    # k = range from 5 to 140, steps of 5
    # distances = all languages in processed GB
    df = pd.read_csv(args.results_file)

    # pretty label for the legend
    df = df.rename({"method": "Method"}, axis=1)

    df["Method"] = df["Method"].map(
        {
            "mmdp": "MaxMin",
            "mdp": "MaxSum",
            "random_genus": "RandomGenus*",
            "random_family": "RandomFamily*",
            "convenience": "Convenience",
            "random": "Random*",
        }
    )

    rand_methods = ["Convenience", "RandomFamily*", "RandomGenus*", "Random*"]
    legend_order = [
        "MaxMin",
        "MaxSum",
        "Convenience",
        "Random*",
        "RandomFamily*",
        "RandomGenus*",
    ]

    OPACITY = 0.7
    COLORS = ["steelblue", "#7D3C98", "chartreuse", "#F4D03F", "red", "#D35400"]
    Y_LABELS = {
        "entropy_with_missing": "Entropy (H)",
        "entropy_without_missing": "Entropy (H)",
        "fvi": "FVI",
        "fvo": "FVO",
        "mpd": "MPD",
    }
    METRICS = [
        # "entropy_with_missing",
        "entropy_without_missing",
        "fvi",
        "fvo",
        "mpd",
    ]

    plots = []
    for metric in METRICS:
        legend = alt.Legend(
            orient="none",
            legendX=130,
            legendY=-40,
            direction="horizontal",
            titleAnchor="middle",
        )

        err_bars = (
            alt.Chart(df[(df["Method"].isin(rand_methods))])
            .mark_errorbar(extent="stdev", opacity=OPACITY)
            # .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("k", title="Sample size"),
                y=alt.Y(metric, title=Y_LABELS[metric]),
                color=alt.Color("Method", legend=legend, sort=legend_order).scale(
                    range=COLORS
                ),
            )
        )

        points = (
            alt.Chart(df)
            .mark_point(filled=True, opacity=OPACITY)
            .encode(
                x=alt.X("k", title="Sample size"),
                y=alt.Y(f"mean({metric})", title=Y_LABELS[metric]),
                color=alt.Color("Method", legend=legend, sort=legend_order).scale(
                    range=COLORS
                ),
            )
        )
        plots.append(err_bars + points)

    top = plots.pop() | plots.pop()
    bottom = plots.pop() | plots.pop()

    combined = alt.vconcat(top, bottom)

    combined.save(f"plots/intrinsic-eval-vis.pdf")


if __name__ == "__main__":
    main()
