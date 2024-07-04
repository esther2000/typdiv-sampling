import json
import pandas as pd

"""
Description: Retrieve intersection of glottocodes from the frames of all sampling methods
"""


# random, maxmin, maxsum: all langs in preprocessed grambank dataframe
gb_cropped_frame = set(pd.read_csv("../../data/gb_processed.csv")["Lang_ID"].to_list())

# convenience: convenience frame
with open("../../data/convenience/convenience_counts.json") as f:
    convenience_frame = set(json.load(f).keys())

# family and genus:
wals_df = pd.read_csv("../../data/wals_dedup.csv")
wals_frame = set(wals_df["Glottocode"].to_list())

int_frame = convenience_frame & wals_frame & gb_cropped_frame

with open("intersection-frame.txt", 'w') as outfile:
    for l in int_frame:
        outfile.write(l+"\n")