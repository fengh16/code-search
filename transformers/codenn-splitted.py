#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse

from common.save_codenn import save_codenn_dataset

def process(input, output, max_vocab_size):
    if not os.path.exists(output):
        os.makedirs(output)
    train = pd.read_csv(os.path.join(input, 'train.csv'))
    validate = pd.read_csv(os.path.join(input, 'valid.csv'))
    test = pd.read_csv(os.path.join(input, 'test.csv'))
    total = pd.read_csv(os.path.join(input, 'total.csv'))
    total.replace('', np.nan, inplace=True)
    total.desc.replace(np.nan, '', inplace=True)
    train, validate, test, total = train.dropna(), \
        validate.dropna(), test.dropna(), total.dropna()
    statistics = save_codenn_dataset(train, test, validate, total,
        max_vocab_size, output)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))
    train[['repo', 'path', 'start', 'url', 'code']] \
        .to_csv(os.path.join(output, 'use.codemap.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    parser.add_argument('--max_vocab_size', help='max vocabulary size',
                        type=int, default=10000)
    args = parser.parse_args()
    process(args.input, args.output, args.max_vocab_size)
