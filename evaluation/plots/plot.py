"""
Note: this code is copied from plot.ipynb to provide a non-notebook version of the code.
"""

import argparse
from pathlib import Path

import altair as alt
import pandas as pd

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
            "maxmin": "MaxMin",
            "maxsum": "MaxSum",
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

    OPACITY = 0.8
    # Colors from Tol's scheme: https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
    COLORS = ["#332288", "#DDCC77", "#CC6677", "#117733", "#88CCEE", "#882255"]
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
                y=alt.Y(metric, title=Y_LABELS[metric], scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Method", legend=legend, sort=legend_order).scale(range=COLORS),
            )
        )

        points = (
            alt.Chart(df)
            .mark_point(filled=True, opacity=OPACITY)
            .encode(
                x=alt.X("k", title="Sample size"),
                y=alt.Y(
                    f"mean({metric})",
                    title=Y_LABELS[metric],
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color("Method", legend=legend, sort=legend_order).scale(range=COLORS),
            )
        )
        plots.append(err_bars + points)

    top = plots.pop() | plots.pop()
    bottom = plots.pop() | plots.pop()

    combined = alt.vconcat(top, bottom)

    combined.save(args.outfile)


if __name__ == "__main__":
    main()
