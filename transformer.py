#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import json
import subprocess

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
        transformers_dir = config['transformers-dir']
        data_cleaned_dir = config['data-cleaned-dir']
        data_transformed_dir = config['data-transformed-dir']

    transformers = {}
    for file in os.listdir(transformers_dir):
        full_path = os.path.join(transformers_dir, file)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            transformers[os.path.splitext(file)[0]] = full_path

    dataset = {}
    for file in os.listdir(data_cleaned_dir):
        name, ext = os.path.splitext(file)
        full_path = os.path.join(data_cleaned_dir, file)
        if os.path.isfile(full_path) and ext =='.csv':
            dataset[name] = full_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer', choices=list(transformers.keys()),
                        help='transformer to use', required=True)
    parser.add_argument('--dataset', choices=list(dataset.keys()),
                        help='dataset to translate to model input', required=True)
    parser.add_argument('--max_vocab_size', help='max vocabulary size for codenn',
                        type=int, default=10000)
    args = parser.parse_args()

    subprocess.run([transformers[args.transformer], dataset[args.dataset],
                    os.path.join(data_transformed_dir, args.dataset),
                   '--max_vocab_size=%d' % args.max_vocab_size])
