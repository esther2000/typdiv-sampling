import pandas as pd
from pathlib import Path
import altair as alt
import argparse

CWD = Path(__file__).parent


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results_file",
        type=Path,
        default="results/experiment-5-500-5-both-metrics.csv",
        help="File with results for sampling method, metrics and sample size.",
    )
    parser.add_argument(
        "-0",
        "--outfile",
        type=Path,
        default="plots/experiment-gb-500.pdf",
        help="Path and filename for output graph.",
    )
    return parser.parse_args()


def main():

    args = create_arg_parser()

    df = pd.read_csv(args.results_file)

    plot = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "k",
                title="Sample size",
            ),
            y=alt.Y(
                "entropy_with_missing",
                # scale=alt.Scale(domain=[1.0, 1.14]),
                title="Entropy",
            ),
            color=alt.Color('method').scale(range=['steelblue', '#7D3C98', 'chartreuse', '#F4D03F', 'red', '#D35400', ])
        )
        .configure_axis(labelFontSize=12, labelFont="serif", titleFontSize=16, titleFont="serif")
    )

    plot.save(args.outfile)


if __name__ == '__main__':
    main()
