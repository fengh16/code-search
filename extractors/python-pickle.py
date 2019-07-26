#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import json
from tqdm import tqdm

from common.extract_codenn import extract

tqdm.pandas()

def main(input, output):
    data = pd.read_pickle(input)
    data.rename(columns={
            'nwo': 'repo',
            'lineno': 'start',
            'original_function': 'code'
        }, inplace=True)
    data = data[['repo', 'path', 'start', 'url', 'code']]
    def extract_content(content):
        try:
            name, start, end, api, token, desc, imported = extract(content)[0]
        except SyntaxError:
            return '', [], set(), ''
        return name, api, token, desc
    data['name'], data['api'], data['token'], data['desc'] = \
        zip(*data.code.progress_map(extract_content))
    data['api'] = data.api.map(lambda api: '|'.join(api))
    data['token'] = data.token.map(lambda token: '|'.join(token))
    data['desc'] = data.desc.map(lambda desc: json.dumps([desc] if desc else []))
    data.to_pickle(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input pickle file contains raw code')
    parser.add_argument('output', help='output pickle file')
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output)):
        os.mkdir(os.path.dirname(args.output))
    main(args.input, args.output)
