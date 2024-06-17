#!/bin/sh

set -xe

# Grambank: download, pre-process, store default same as args
# assume gb_lang_feat_vals.csv is here already, maybe pull from github and create? --> there actually is a separate python script for this
python data/compute_all_distances.py -b -n -g data/gb_lang_feat_vals.csv -o data/bin-norm-distances.csv

# WALS: download, pre-process, store default


# TODO:
# - add glottolog --> languoid.csv in DATA folder: https://cdstar.eva.mpg.de//bitstreams/EAEA0-CFBC-1A89-0C8C-0/glottolog_languoid.csv.zip
# - add annotations for convenience baseline: https://github.com/WPoelman/typ-div-survey/blob/master/data/annotations-enhanced.csv
