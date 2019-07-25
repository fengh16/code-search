#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tables
import os
import sys
import pickle
import argparse
import json
from sklearn import model_selection

def build_vocab(series, max_vocab_size):
    vocab_num = {}
    series = list(map(lambda x: x.split('|'), list(series)))
    for i in series:
        for j in i:
            vocab_num[j] = vocab_num.get(j, 0) + 1
    origin_vocab_size = len(vocab_num)
    vocab_rev = list(vocab_num.items())
    vocab_rev.sort(key=lambda x: x[1], reverse=True)
    vocab_rev = list(map(lambda x: x[0], vocab_rev))
    vocab_rev = vocab_rev[:max_vocab_size - 2]
    # 0: padding; 1: unknown; 2...: vocabulary
    return ({v: k + 2 for k, v in enumerate(vocab_rev)},
            ['<PAD>', '<UNK>'] + vocab_rev, origin_vocab_size)

def save_to_h5(series, vocab, file):
    with tables.open_file(file, mode='w') as h5f:
        table = h5f.create_table('/', 'indices', {
            'length': tables.UInt32Col(),
            'pos': tables.UInt32Col()
        }, 'a table of indices and lengths')
        array = h5f.create_earray('/', 'phrases', tables.Int16Atom(), (0,))
        array.flavor = 'numpy'
        pos = 0
        for item in series:
            item = item.split('|')
            length = len(item)
            index = table.row
            index['length'] = length
            index['pos'] = pos
            index.append()
            array.append(list(map(lambda x: vocab.get(x, 1), item)))
            pos += length

def process(input, output, max_vocab_size):
    if not os.path.exists(output):
        os.makedirs(output)
    origin_data = pd.read_csv(input)
    origin_data.replace(np.nan, '', inplace=True)
    data = origin_data.replace('', np.nan)
    data = data.dropna()
    train, test = model_selection.train_test_split(data, test_size=0.2)
    validate, test = model_selection.train_test_split(test, test_size=0.5)
    series = [
        ('methname', lambda x: x.name),
        ('apiseq', lambda x: x.api),
        ('tokens', lambda x: x.token),
        ('desc', lambda x: x.desc)
    ]
    tasks = [('train', train), ('valid', validate), ('test', test), ('use', origin_data)]
    vocabs = []
    statistics = {}
    for name, select in series:
        vocab, vocab_rev, origin_vocab_size = build_vocab(select(origin_data), max_vocab_size)
        statistics['%sOriginVocabSize' % name] = origin_vocab_size
        statistics['%sVocabSize' % name] = len(vocab_rev)
        print('%s origin vocab size: %d' % (name, origin_vocab_size))
        print('%s processed vocab size: %d' % (name, len(vocab_rev)))
        with open(os.path.join(output, 'vocab.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(output, 'vocabrev.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab_rev, f)
        vocabs.append(vocab)
    for task_name, task in tasks:
        statistics['%sDatasetSize' % task_name] = task.shape[0]
        print('%s set size: %d' % (task_name, task.shape[0]))
        for (series_name, select), vocab in zip(series, vocabs):
            if task_name == 'use' and series_name == 'desc':
                continue
            save_to_h5(select(task), vocab, os.path.join(output,
                '%s.%s.h5' % (task_name, series_name)))
    with open(os.path.join(output, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
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
