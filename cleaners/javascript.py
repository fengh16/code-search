#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import itertools
import spacy
import sys
import argparse
import json

tqdm.pandas()
split_name_regex = re.compile(r'(.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$))')

re_remove = {
  "discard_remain": [re.compile(r, re.IGNORECASE) for r in [
    "^(//)?(\\s)*todo",
    "(//)?(\\s)*['\"]use(\\s)+strict['\"]",
    "(//)?(\\s)*eslint-disable",
    "^(//)?(\\s)*export",
    "(//)?(\\s)*var ",
    "(.*)this code is generated(.*)"
  ]],
  "discard_line": [re.compile(r, re.IGNORECASE) for r in [
    "^(//)?(\\s)*@"
  ]],
  "is_ascii": re.compile("^[\x00-\x7F]*$")
}

def split_name(name):
    matches = re.finditer(split_name_regex, name)
    return list(itertools.chain(
        *[[i for i in m.group(0).split('_') if i] for m in matches]))


def remove_useless_comments(comments):
    ans = []
    for comment in comments:
        for c in comment.split('\n'):
            need_break = False
            need_continue = False
            for a in re_remove["discard_remain"]:
                if a.match(c):
                    need_break = True
                    break
            if need_break:
                break
            for a in re_remove["discard_line"]:
                if a.match(c):
                    need_continue = True
                    break
            if need_continue or not re_remove["is_ascii"].match(c):
                continue
            ans.append(c.strip())
    return json.dumps('\n'.join(ans))


def main(input, output):
    data = pd.read_csv(input)
    data.replace('', np.nan, inplace=True)
    data.desc.replace(np.nan, '', inplace=True)
    data.imported.replace(np.nan, '', inplace=True)
    data = data.dropna().drop_duplicates()
    data.name = data.name.str.replace('$', '_')
    data = data[~data.name.str.startswith('_') & (data.name.str.len() > 2)]
    data.name = data.name.apply(lambda x: '|'.join(
        [i.lower() for i in split_name(x) if not i.isdigit()]))
    data.api = data.api.apply(lambda x: '|'.join(
        [i.split('.')[-1] for i in x.split('|')]))
    data.token = data.token.str.replace('$', '_')
    data.token = data.token.apply(lambda x: '|'.join(
        list(set(itertools.chain(
            *[[i.lower() for i in split_name(y) if not i.isdigit()]
                for y in x.split('|')])))))
    data.desc = data.desc.apply(lambda x:
        re.sub(r'(\*| |\n)+', ' ', next(filter(None, x.split('\n\n')), '')).strip())
    data.desc = data.desc.apply(lambda x: remove_useless_comments(json.loads(x)))
    nlp = spacy.load('en_core_web_lg')
    data.desc = data.desc.progress_apply(lambda x:
        '|'.join([token.lemma_ for token in nlp('\n'.join(json.loads(x)))
                  if token.is_alpha and not token.is_stop]))
    data.replace('', np.nan, inplace=True)
    data.desc.replace(np.nan, '', inplace=True)
    data = data.dropna()
    data.to_csv(output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input csv that contains origin data')
    parser.add_argument('output', help='output csv that cleaned data is written to')
    args = parser.parse_args()
    main(args.input, args.output)
