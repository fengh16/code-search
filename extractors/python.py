#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import csv
from os import path
import argparse
from tqdm import tqdm
import json

from common.extract_codenn import extract

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path that contains raw code')
    parser.add_argument('output', help='output csv that extracted data is written to')
    args = parser.parse_args()
    files = glob.glob(path.join(args.input, '**/*.py'), recursive=True)
    with open(args.output, "w") as out:
        f = csv.writer(out)
        f.writerow(['file', 'start', 'end', 'name', 'api', 'token', 'desc', 'imported'])
        for file in tqdm(files):
            if not path.isfile(file):
                continue
            try:
                with open(file) as source:
                    content = source.read()
                feature = extract(content)
            except:
                continue
            for item in feature:
                name, start, end, api, token, desc, imported = item
                f.writerow((file, start, end, name,
                            '|'.join(api), '|'.join(token),
                            json.dumps([desc] if desc else []), '|'.join(imported)))
