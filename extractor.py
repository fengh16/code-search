#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import json
import subprocess

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
        extractor_dir = config['extractors-dir']
        data_rawcode_dir = config['data-rawcode-dir']
        data_extracted_dir = config['data-extracted-dir']

    extractors = {}
    for file in os.listdir(extractor_dir):
        extractors[os.path.splitext(file)[0]] = os.path.join(extractor_dir, file)

    dataset = {}
    for rawcode in os.listdir(data_rawcode_dir):
        dataset[rawcode] = os.path.join(data_rawcode_dir, rawcode)

    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', choices=list(extractors.keys()),
                        help='language of the extractor', required=True)
    parser.add_argument('--dataset', choices=list(dataset.keys()),
                        help='dataset to extract from', required=True)
    args = parser.parse_args()

    subprocess.run([extractors[args.extractor], dataset[args.dataset],
                    os.path.join(data_extracted_dir, args.dataset + '.csv')])