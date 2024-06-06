#!/usr/bin/env python3
# coding=utf-8
# Copyright  2022  Ruiqi WANG, Jinpeng LI, Jiayu DU
#
# only tested and validated on pynini v2.1.5 via : 'conda install -c conda-forge pynini'
# pynini v2.1.0 doesn't work
#

import sys, os, argparse
import string
from nemo_text_processing.text_normalization.normalize import Normalizer

def read_interjections(filepath):
    interjections = []
    with open(filepath) as f:
        for line in f:
            words = [ x.strip() for x in line.split(',') ]
            interjections += [ w for w in words ] + [ w.upper() for w in words ] + [ w.lower() for w in words ]
    return list(set(interjections))  # deduplicated


nemo_tn_en = Normalizer(input_case='lower_cased', lang='en')

itj = read_interjections(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interjections_en.csv')
)
itj_map = { x : True for x in itj }

certain_single_quote_items = ["\"'", "'?", "'!","'.", "?'", "!'", ".'","''", "<BOS>'", "'<EOS>"]
single_quote_removed_items = [ x.replace("'", '') for x in certain_single_quote_items ]

puncts_to_remove = string.punctuation.replace("'", '')+"—–“”"
puncts_trans = str.maketrans(puncts_to_remove, ' ' * len(puncts_to_remove), '')


class EnglishNormalizer(object):

    def __init__(self):
        super().__init__()
    
    def __call__(self, text):
        text = text.replace("‘","'").replace("’","'")

        # nemo text normalization
        # modifications to NeMo:
        # 1. added UK to US conversion: nemo_text_processing/text_normalization/en/data/whitelist/UK_to_US.tsv
        # 2. swith 'oh' to 'o' in year TN to avoid confusion with interjections, e.g.:
        #    1805: eighteen oh five -> eighteen o five
        text = nemo_tn_en.normalize( text.lower() )

        # Punctuations
        # NOTE(2022.10 Jiayu):
        # Single quote removal is not perfect.
        # ' needs to be reserved for:
        #     Abbreviations:
        #       I'm, don't, she'd, 'cause, Sweet Child o' Mine, Guns N' Roses, ...
        #     Possessions:
        #       John's, the king's, parents', ...
        text = '<BOS>' + text + '<EOS>'
        for x, y in zip(certain_single_quote_items, single_quote_removed_items):
            text = text.replace(x, y)
        text = text.replace('<BOS>','').replace('<EOS>','')

        text = text.translate(puncts_trans).replace(" ' "," ")

        # Interjections
        text = ' '.join([ x for x in text.strip().split() if x not in itj_map ])
        text = text.lower()

        return text


if __name__ == '__main__':
    normalizer = EnglishNormalizer()
    text = "i've been going to beijing. since 2010s. I think"
    print(normalizer(text))