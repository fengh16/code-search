import os
import numpy as np
import pandas as pd
import pickle
import json
from keras_preprocessing.sequence import pad_sequences

from .preprocessing import ktext_process_text
from .build_vocab import build_vocab, heuristic_seq_len

default_config = {
    'token': {
        'append_indicators': False,
        'max_vocab_size': 20000,
        'heuristic_percent': 0.7,
        'padding': 'pre',
        'truncating': 'post'
    },
    'desc': {
        'append_indicators': True,
        'max_vocab_size': 14000,
        'heuristic_percent': 0.7,
        'padding': 'post',
        'truncating': 'post'
    }
}

def save_code_summarizer_dataset(input, output, config=default_config):
    series = ['token', 'desc']
    tasks = ['train', 'valid', 'test', 'total']
    statistics = {}
    vocabs = []
    sequence_length = []
    for name in series:
        origin_series = pd.read_csv(os.path.join(input, 'total.%s.csv' % name),
                                    header=None, names=[name])[name]
        origin_series.replace()
        origin_series.dropna(inplace=True)
        tokenized_series = ktext_process_text(origin_series,
            append_indicators=config[name]['append_indicators'])
        word2code, code2word, origin_vocab_size = build_vocab(tokenized_series,
            max_vocab_size=config[name]['max_vocab_size'])
        statistics['%sOriginVocabSize' % name] = origin_vocab_size
        statistics['%sVocabSize' % name] = len(code2word)
        with open(os.path.join(output, 'word2code.%s.pkl' % name), 'wb') as f:
            pickle.dump(word2code, f)
        with open(os.path.join(output, 'code2word.%s.pkl' % name), 'wb') as f:
            pickle.dump(code2word, f)
        vocabs.append(word2code)
        length = heuristic_seq_len(tokenized_series, config[name]['heuristic_percent'])
        statistics['%sSequenceLength' % name] = length
        sequence_length.append(length)
    for task_name in tasks:
        for series_name, word2code, length in zip(series, vocabs, sequence_length):
            if task_name == 'total' and series_name == 'desc':
                continue
            origin_series = pd.read_csv(os.path.join(input, '%s.%s.csv' %
                (task_name, series_name)), header=None, names=[series_name])[series_name]
            data_size_key = '%sDatasetSize' % task_name
            if data_size_key in statistics:
                assert statistics[data_size_key] == origin_series.size, 'Corrupted data'
            else:
                statistics[data_size_key] = origin_series.size
            tokenized_series = ktext_process_text(origin_series,
                append_indicators=config[series_name]['append_indicators'])
            encoded_series = tokenized_series.map(lambda x: [word2code.get(i, 1) for i in x])
            padded_series = pad_sequences(encoded_series, length, 'int32',
                config[series_name]['padding'], config[series_name]['truncating'], 0)
            padded_series = np.array(padded_series)
            np.save(os.path.join(output, '%s.%s.npy' % (task_name, series_name)),
                    padded_series)
    with open(os.path.join(output, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)
    return statistics
