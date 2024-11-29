#!/bin/sh

set -xe

# Grambank: download, pre-process, store default same as args
# assume gb_lang_feat_vals.csv is here already, maybe pull from github and create? --> there actually is a separate python script for this
# assume wals is here already as well

cd data

# Glottolog
wget https://cdstar.eva.mpg.de//bitstreams/EAEA0-CFBC-1A89-0C8C-0/glottolog_languoid.csv.zip
unzip glottolog_languoid.csv.zip
rm README.txt glottolog_languoid.csv.zip

# Annotations for convenience baseline
wget https://raw.githubusercontent.com/WPoelman/typ-div-survey/master/data/annotations-enhanced.csv

cd ..

python data/compute_all_distances.py -b -n -c 0.25 -r
