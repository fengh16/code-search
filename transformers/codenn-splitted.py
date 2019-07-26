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
    train, validate, test = train.dropna(), validate.dropna(), test.dropna()
    origin_data = pd.concat((train, validate, test))
    statistics = save_codenn_dataset(train, test, validate, origin_data,
        max_vocab_size, output)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))
    origin_data.rename(columns={'original_function': 'code'}, inplace=True)
    origin_data[['code']].to_csv(os.path.join(output, 'use.codemap.csv'),
                                 index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    parser.add_argument('--max_vocab_size', help='max vocabulary size',
                        type=int, default=10000)
    args = parser.parse_args()
    process(args.input, args.output, args.max_vocab_size)
