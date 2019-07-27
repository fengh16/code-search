#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse
from sklearn import model_selection

from common.save_codenn import save_codenn_dataset

def process(input, output, max_vocab_size):
    if not os.path.exists(output):
        os.makedirs(output)
    origin_data = pd.read_csv(input)
    origin_data.replace(np.nan, '', inplace=True)
    data = origin_data.replace('', np.nan)
    data = data.dropna()
    train, test = model_selection.train_test_split(data, test_size=0.2)
    validate, test = model_selection.train_test_split(test, test_size=0.5)
    statistics = save_codenn_dataset(train, validate, test, origin_data,
        max_vocab_size, output)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))
    origin_data[['file', 'start', 'end']].to_csv(os.path.join(output, 'use.codemap.csv'),
                                                 index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    parser.add_argument('--max_vocab_size', help='max vocabulary size',
                        type=int, default=10000)
    args = parser.parse_args()
    process(args.input, args.output, args.max_vocab_size)
