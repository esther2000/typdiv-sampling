#!/bin/sh

set -xe

# Grambank: download, pre-process, store default same as args
# assume gb_lang_feat_vals.csv is here already, maybe pull from github and create?
python data/compute_all_distances.py -b -n -g data/gb_lang_feat_vals.csv -o data/bin-norm-distances.csv
# Uriel: download, store default

# WALS: download, pre-process, store default

