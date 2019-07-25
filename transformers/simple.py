#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import json
from sklearn import model_selection

def main(input, output, word2vec):
    if not os.path.exists(output):
        os.makedirs(output)
    with open(word2vec, 'rb') as f:
        vocabrev, _ = pickle.load(f, encoding='iso-8859-1')
    vocab = {v: k for k, v in enumerate(vocab_rev)}
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
    statistics = {}
    for task_name, task in tasks:
        statistics['%sDatasetSize' % task_name] = task.shape[0]
        print('%s set size: %d' % (task_name, task.shape[0]))
        vocab = {v: k for k, v in enumerate(vocab_rev)}
    with open(os.path.join(output, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    parser.add_argument('--word2vec', help='path to word2vec model pickle', required=True)
    args = parser.parse_args()
    main(args.input, args.output, args.max_vocab_size)
