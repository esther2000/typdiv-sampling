#!/bin/bash

set -xe

# Compute language distances, with the following options:
# binarize multistate values, remove macro-languages, crop languages for which >25% data is unavailable, normalize distances
python ../data/prepare_grambank.py -b -n -r -c 0.25 \
    -g ../data/gb_lang_feat_vals.csv \
    -d ../data/new_dists.csv \
    -o ../data/gb_processed.csv

# Sample with all methods and various values
python experiment.py \
    --results_path results/intrinsic-eval-refactor.csv \
    --dist_path ../data/new_dists.csv \
    -gb_path ../grambank/cldf/languages.csv \
    -wals_path ../data/wals_dedup.csv \
    -gb_features_path ../data/gb_processed.csv \
    -counts_path ../data/convenience/convenience_counts.json \
    -rand_runs 10 \
    -s 2 \
    -e 140 \
    -st 3

# Visualize results in plots
python plots/plot.py -r results/intrinsic-eval-refactor.csv -o plots/intrinsic-eval-refactor.pdf
