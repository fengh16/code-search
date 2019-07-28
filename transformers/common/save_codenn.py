import tables
import pickle
import json
import os
from .build_vocab import build_vocab

def save_codenn_series(series, vocab, file, unknown_idx=1):
    with tables.open_file(file, mode='w') as h5f:
        table = h5f.create_table('/', 'indices', {
            'length': tables.UInt32Col(),
            'pos': tables.UInt32Col()
        }, 'a table of indices and lengths')
        array = h5f.create_earray('/', 'phrases', tables.Int32Atom(), (0,))
        array.flavor = 'numpy'
        pos = 0
        for item in series:
            item = item.split('|')
            length = len(item)
            index = table.row
            index['length'] = length
            index['pos'] = pos
            index.append()
            array.append(list(map(lambda x: vocab.get(x, unknown_idx), item)))
            pos += length

def save_codenn_dataset(train, validate, test, origin_data,
                        max_vocab_size, output_path, unknown_idx=1):
    series = [
        ('methname', 'name'),
        ('apiseq', 'api'),
        ('tokens', 'token'),
        ('desc', 'desc')
    ]
    tasks = [('train', train), ('valid', validate), ('test', test), ('use', origin_data)]
    vocabs = []
    statistics = {}
    for name, select in series:
        data = list(map(lambda x: x.split('|'), list(origin_data[select])))
        vocab, vocab_rev, origin_vocab_size = build_vocab(data, max_vocab_size)
        statistics['%sOriginVocabSize' % name] = origin_vocab_size
        statistics['%sVocabSize' % name] = len(vocab_rev)
        with open(os.path.join(output_path, 'vocab.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab, f)
        with open(os.path.join(output_path, 'vocabrev.%s.pkl' % name), 'wb') as f:
            pickle.dump(vocab_rev, f)
        vocabs.append(vocab)
    for task_name, task in tasks:
        statistics['%sDatasetSize' % task_name] = task.shape[0]
        for (series_name, select), vocab in zip(series, vocabs):
            if task_name == 'use' and series_name == 'desc':
                continue
            save_codenn_series(task[select], vocab, os.path.join(output_path,
                '%s.%s.h5' % (task_name, series_name)), unknown_idx=unknown_idx)
    with open(os.path.join(output_path, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    return statistics
