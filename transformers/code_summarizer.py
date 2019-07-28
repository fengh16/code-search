#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

from common.save_code_summarizer import save_code_summarizer_dataset

def main(input, output):
    if not os.path.exists(output):
        os.makedirs(output)
    statistics = save_code_summarizer_dataset(input, output)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv that contains cleaned data')
    parser.add_argument('output', help='output path that dataset should be generated into')
    args = parser.parse_args()
    main(args.input, args.output)
