#!/bin/bash

#set -xe
#pip install -e .

# Compute language distances, with the following options:
# (binarize multistate values, remove macro-languages, crop languages for which >25% data is unavailable, normalize distances)
python ../data/compute_all_distances.py -b -n -r -c 0.75 \
                                     -g ../data/gb_lang_feat_vals.csv \
                                     -o ../data/gb_lang_dists-bnrc75.csv

# Sample with all methods and various values
python experiment.py \
    --results_path results/intrinsic-eval.csv \
    --dist_path ../data/gb_lang_dists-bnrc75.csv \
    -gb_path ../grambank/cldf/languages.csv \
    -wals_path ../data/wals_dedup.csv \
    -gb_features_path ../data/gb_processed.csv \
    -counts_path ../data/convenience/convenience_counts.json \
    -rand_runs 10 \
    -s 10 \
    -e 140 \
    -st 5

# Visualize results in plots
python plots/plot.py -r results/intrinsic-eval.csv -o plots/intrinsic-eval-vis.pdf