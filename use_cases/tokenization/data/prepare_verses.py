import os
from collections import defaultdict, Counter
import random
import json

BIBLE_PATH = '/Users/jk31ds/Desktop/bible_corpus'
LANG_CODES = 'langcodes_Latn.txt'
C_SIZE = 2000
TEST_SIZE = 1000


def get_verse_ids(filename):
    """ Given a bible file, return the verse IDs that it contains. """
    verses = set()
    with open(os.path.join(BIBLE_PATH, filename), 'r') as bib:
        lines = bib.readlines()
    for line in lines[11:]:
        verses.add(line.split()[0].replace('#', ''))

    return verses


def get_verse_text(filename, verse_id):
    """ Return the text of a verse, given its bible file and verse ID"""
    with open(os.path.join(BIBLE_PATH, filename), 'r') as bib:
        lines = bib.readlines()

    for line in lines[11:]:
        if line.split()[0].replace('#', '') == verse_id:
            return ' '.join(line.split()[1:])


def bible_to_dict(biblefile):
    bible_dict = dict()
    with open(f'{BIBLE_PATH}/{biblefile}', "r") as bf:
        lines = bf.readlines()

    for line in lines[11:]:
        v_id = line.split()[0]
        v_text = ' '.join(line.split()[1:])
        bible_dict[v_id] = v_text

    return bible_dict


def main():

    # Get lang codes (filtered with fam/typ coverage)
    with open(LANG_CODES, 'r') as infile:
        codes = [x.strip() for x in infile]

    # Get all bibles per lang
    bib_per_lang = defaultdict(list)
    for root, dirs, files in os.walk(BIBLE_PATH, topdown=False):
        for name in files:
            if name[:3] in codes:
                bib_per_lang[name[:3]].append(name)

    # Select the bible per lang with most verses
    max_len_bib = dict()
    for k, v in bib_per_lang.items():
        if len(v) > 1:
            max_len = (0,)
            for bibname in v:
                length = len(get_verse_ids(bibname))
                if length > max_len[0]:
                    max_len = (length, bibname)
            max_len_bib[k] = max_len[1]
        else:
            max_len_bib[k] = v[0]

    # Parallel verses between all bibles
    c = Counter()
    filenames = [x for _, x in max_len_bib.items()]
    for f in filenames:
        c.update(get_verse_ids(f))

    # Remove languages that do not have coverage for each of the most commonly translated verses
    mcv = set([x[0] for x in c.most_common(C_SIZE)])
    langs = []
    for k, v in max_len_bib.items():
        verses = get_verse_ids(v)
        if len(verses.intersection(mcv)) == len(mcv):
            langs.append(k)
    bibdict = {l: max_len_bib[l] for l in langs}

    # Make directories
    os.mkdir('verses')

    # Split test and valid (randomly from most common)
    random.seed(0)
    test_verse_ids = set(random.sample(list(mcv), TEST_SIZE))

    n = -1
    lang_set = set(langs)
    for lang in lang_set:
        n+=1
        print('Working on: ', lang, n)
        biblefile = bibdict[lang]
        verse_dict = bible_to_dict(biblefile)

        # write verses
        outfile_lang_valid = f'verses/{lang}.txt'
        with open(outfile_lang_valid, 'w') as out_lang:
            for v_id in test_verse_ids:
                v_text = verse_dict.get(v_id)
                out_lang.write(v_text+'\n')


if __name__ == '__main__':
    main()
