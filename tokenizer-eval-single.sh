#!/bin/sh

set -xe

# pip install -e .
# python data/compute_all_distances.py -b -n -g data/gb_lang_feat_vals.csv -o data/bin-norm-distances.csv
python evaluation/experiment.py \
    --results_path evaluation/experiments/tokenizer-eval-equal-lang.csv \
    --frame data/frames/equal_lang_frame.txt \
    --dist_path data/bin-norm-distances.csv \
    -gb_path grambank/cldf/languages.csv \
    -wals_path data/wals_dedup.csv \
    -gb_features_path data/gb_binarized.csv \
    -counts_path data/convenience_counts.json \
    -rand_runs 10 \
    -s 3 \
    -e 89 \
    -st 1
