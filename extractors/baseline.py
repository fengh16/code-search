#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle

def save_to_csvs(df, name, path):
    df.function_tokens.to_csv(os.path.join(path, '%s.token.csv' % name),
                              index=False, header=False)
    df.original_function.to_json(os.path.join(path, '%s.code.json.gz' % name),
                                 orient='values', compression='gzip')
    df.docstring_tokens.to_csv(os.path.join(path, '%s.desc.csv' % name),
                                index=False, header=False)
    df.url.to_csv(os.path.join(path, '%s.url.csv' % name),
                  index=False, header=False)

def main(input, output):
    if not os.path.exists(output):
        os.makedirs(output)
    for task in ['train', 'valid', 'test', 'total']:
        with open(os.path.join(input, '%s.pkl' % task), 'rb') as f:
            data = pickle.load(f)
        save_to_csvs(data, task, output)
    save_to_csvs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path that contains raw code pickle')
    parser.add_argument('output', help='output path that dataset should be generated into')
    args = parser.parse_args()
    main(args.input, args.output)
