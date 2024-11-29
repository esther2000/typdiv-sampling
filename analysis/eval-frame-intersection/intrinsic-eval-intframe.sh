#!/bin/bash

#set -xe
#pip install -e .

# Get intersection of sampling frame for all languages
python frame_intersection.py


# Compute language distances, with the following options:
# binarize multistate values, remove macro-languages, crop languages for which >25% data is unavailable, normalize distances
python ../../data/compute_all_distances.py -b -n -r -c 0.75 \
                                     -g ../../data/gb_lang_feat_vals.csv \
                                     -o gb_lang_dists-bnrc75-intframe.csv \
                                     -l intersection-frame.txt

# Sample with all methods and various values
python ../../evaluation/experiment.py \
    --results_path intrinsic-eval-intframe.csv \
    --dist_path gb_lang_dists-bnrc75-intframe.csv  \
    -gb_path ../../grambank/cldf/languages.csv \
    -wals_path ../../data/wals_dedup.csv \
    -gb_features_path ../../data/gb_processed.csv \
    -counts_path ../../data/convenience/convenience_counts.json \
    -rand_runs 10 \
    -s 2 \
    -e 132 \
    -st 3


# Visualize results in plots
python ../../evaluation/plots/plot.py -r intrinsic-eval-intframe.csv -o intrinsic-eval-int.pdf
