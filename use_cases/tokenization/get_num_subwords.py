import argparse
import pandas as pd
from tokenizers import Tokenizer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained tokenizer from HuggingFace.",
    )
    parser.add_argument(
        "-v",
        "--verses",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="avg_num_subwords.csv",
        help="File to write output to.",
        type=str
    )

    return parser.parse_args()


def remove_spec(tokens):
    return [t for t in tokens if not t in ["[SEP]", "[CLS]"]]


def main():
    args = create_arg_parser()

    tok = Tokenizer.from_pretrained(args.tokenizer)

    # load verses
    with open(args.verses, 'r') as v1:
        verses_l1 = [verse.rstrip() for verse in v1]

    # get glottocodes
    iso = args.verses[-7:-4]
    gltc_df = pd.read_csv('../../data/languoid.csv')
    gltc = gltc_df[gltc_df['iso639P3code'] == iso]['id'].to_list()[0]

    # tokenizer
    subwords_l1 = [remove_spec(t.tokens) for t in tok.encode_batch(verses_l1)]

    # calculate premium
    ns = []
    for x in subwords_l1:
        if x:
            ns.append(len(x))

    # write to outfile
    with open(args.outfile, 'a') as outfile:
        outfile.write(f'{args.verses};{sum(ns)/len(ns)};{gltc}\n')


if __name__ == "__main__":
    main()





