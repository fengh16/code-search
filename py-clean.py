import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import itertools
import spacy
import sys

tqdm.pandas()
split_name_regex = re.compile(r'(.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$))')

def split_name(name):
    matches = re.finditer(split_name_regex, name)
    return list(itertools.chain(
        *[[i for i in m.group(0).split('_') if i] for m in matches]))

def main(input, output):
    data = pd.read_csv(input)
    data = data.dropna(how='any',axis=0).drop_duplicates()
    data = data[~data.name.str.startswith('_')]
    data.name = data.name.apply(lambda x: '|'.join(
        [i.lower() for i in split_name(x) if not i.isdigit()]))
    data.api = data.api.apply(lambda x: '|'.join(
        [i.split('.')[-1] for i in x.split('|')]))
    data.token = data.token.apply(lambda x: '|'.join(
        list(set(itertools.chain(
            *[[i.lower() for i in split_name(y) if not i.isdigit()]
                for y in x.split('|')])))))
    data.desc = data.desc.apply(lambda x:
        next(filter(None, x.split('\n\n'))).strip().replace('\n', ' '))
    nlp = spacy.load('en_core_web_lg')
    data.desc = data.desc.progress_apply(lambda x:
        '|'.join([token.lemma_ for token in nlp(x)
                  if token.is_alpha and not token.is_stop]))
    data.replace('', np.nan, inplace=True)
    data = data.dropna(how='any',axis=0)
    data.to_csv(output)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('py-clean.py <input_file> <output_file>')
    else:
        main(sys.argv[1], sys.argv[2])
