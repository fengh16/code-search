#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn import model_selection

from common.save_codenn import save_codenn_dataset_with_vocab

def main(input, output, word2vec):
    if not os.path.exists(output):
        os.makedirs(output)
    with open(word2vec, 'rb') as f:
        vocab_rev, _ = pickle.load(f, encoding='iso-8859-1')
    vocab = {v: k for k, v in enumerate(vocab_rev)}
    origin_data = pd.read_csv(input)
    origin_data.replace(np.nan, '', inplace=True)
    data = origin_data.replace('', np.nan)
    data = data.dropna()
    train, test = model_selection.train_test_split(data, test_size=0.2)
    validate, test = model_selection.train_test_split(test, test_size=0.5)
    statistics = save_codenn_dataset_with_vocab(train, test, validate, origin_data,
        vocab, output, unkown_idx=0)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))
    origin_data[['file', 'start', 'end']].to_csv(os.path.join(output, 'use.codemap.csv'),
                                                 index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    parser.add_argument('--word2vec', help='path to word2vec model pickle', required=True)
    args = parser.parse_args()
    main(args.input, args.output, args.word2vec)
