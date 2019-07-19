import pandas as pd
import numpy as np
import tables
import os
import sys
import pickle
from sklearn import model_selection

def build_vocab(series, num=10000):
    vocab_num = {}
    series = list(map(lambda x: x.split('|'), list(series)))
    for i in series:
        for j in i:
            vocab_num[j] = vocab_num.get(j, 0) + 1
    vocab_rev = list(vocab_num.items())
    vocab_rev.sort(key=lambda x: x[1], reverse=True)
    vocab_rev = list(map(lambda x: x[0], vocab_rev))
    vocab_rev = vocab_rev[:num]
    return {v: k + 1 for k, v in enumerate(vocab_rev)}, vocab_rev

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
            array.append(list(map(lambda x: vocab.get(x, 0), item)))
            pos += length

def process(input, output):
    if not os.path.exists(output):
        os.makedirs(output)
    data = pd.read_csv(input, index_col=0)
    data.replace('', np.nan, inplace=True)
    data = data.dropna(how='any',axis=0)
    train, test = model_selection.train_test_split(data, test_size=0.2)
    validate, test = model_selection.train_test_split(test, test_size=0.5)
    series = [
        ('methname', lambda x: x.name),
        ('apiseq', lambda x: x.api),
        ('tokens', lambda x: x.token),
        ('desc', lambda x: x.desc)
    ]
    tasks = [('train', train), ('valid', validate), ('test', test)]
    vocabs = [];
    for name, select in series:
        vocab, vocab_rev = build_vocab(select(data))
        with open(os.path.join(output, 'vocab.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(output, 'vocabrev.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab_rev, f)
        vocabs.append(vocab)
    for task_name, task in tasks:
        for (series_name, select), vocab in zip(series, vocabs):
            save_to_h5(select(task), vocab, os.path.join(output,
                '%s.%s.h5' % (task_name, series_name)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("transform.py <input_file> <output_path>")
    else:
        process(sys.argv[1], sys.argv[2])
