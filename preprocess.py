#!/usr/bin/python3
"""
Description:    Preprocess grambank and get langs in there (2 things)
Usage:          ...
"""

import argparse
import pandas as pd
from collections import defaultdict


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--gb_folder", type=str,
                        default='sources/grambank-v1.0.3',
                        help="Folder where Grambank download is stored")
    args = parser.parse_args()
    return args


def main():

    args = create_arg_parser()

    # Get Grambank data
    gb_values = pd.read_csv(f'{args.gb_folder}/cldf/values.csv')
    gb_data = gb_values[['Language_ID', 'Parameter_ID', 'Value']]

    # Create ordered lists of lang and feature IDS (unique)
    lang_ids = sorted(list(set(gb_data['Language_ID'].to_list())))
    feat_ids = sorted(list(set(gb_data['Parameter_ID'].to_list())))

    with open(r'data/gb_langs.txt', 'w') as langs:
        langs.write('\n'.join(lang_ids))

    # Make lookup table for some efficiency
    gb_lookup = defaultdict(dict)
    for f_id in feat_ids:
        f_df = gb_data.loc[gb_data['Parameter_ID'] == f_id]
        f_vals = [(feat, val) for feat, val in zip(f_df['Language_ID'].to_list(),f_df['Value'].to_list())]
        for lang, val in f_vals:
            gb_lookup[f_id][lang] = val

    # Initialize DF
    gb_df = pd.DataFrame(columns=['Lang_ID'] + [f for f in feat_ids])
    gb_df['Lang_ID'] = lang_ids

    # Write feature values to DF
    for feat_id in feat_ids:
        print('Working on feature:', feat_id)
        for lang_id in lang_ids:
            try:
                gb_df.loc[gb_df['Lang_ID'] == lang_id , feat_id] = gb_lookup[feat_id][lang_id]
            except KeyError:
                gb_df.loc[gb_df['Lang_ID'] == lang_id, feat_id] = 'no_cov'

    # Write to file
    gb_df.to_csv('data/gb_lang_feat_vals.csv')

    # TODO: add WALS


if __name__ == '__main__':
    main()
