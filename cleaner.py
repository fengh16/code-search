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
        cleaner_dir = config['cleaners-dir']
        data_extracted_dir = config['data-extracted-dir']
        data_cleaned_dir = config['data-cleaned-dir']

    cleaners = {}
    for file in os.listdir(cleaner_dir):
        full_path = os.path.join(cleaner_dir, file)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            cleaners[os.path.splitext(file)[0]] = full_path

    dataset = {}
    for file in os.listdir(data_extracted_dir):
        name, ext = os.path.splitext(file)
        full_path = os.path.join(data_extracted_dir, file)
        if os.path.isfile(full_path) and ext =='.csv':
            dataset[name] = full_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaner', choices=list(cleaners.keys()),
                        help='cleaner to use', default='default')
    parser.add_argument('--dataset', choices=list(dataset.keys()),
                        help='dataset to clean', required=True)
    args = parser.parse_args()

    subprocess.run([cleaners[args.cleaner], dataset[args.dataset],
                    os.path.join(data_cleaned_dir, args.dataset + '.csv')])
