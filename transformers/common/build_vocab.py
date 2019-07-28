from collections import Counter
import pandas as pd

def build_vocab(series, max_vocab_size):
    vocab_num = {}
    for i in series:
        for j in i:
            vocab_num[j] = vocab_num.get(j, 0) + 1
    origin_vocab_size = len(vocab_num)
    vocab_rev = list(vocab_num.items())
    vocab_rev.sort(key=lambda x: x[1], reverse=True)
    vocab_rev = list(map(lambda x: x[0], vocab_rev))
    vocab_rev = vocab_rev[:max_vocab_size - 2]
    # 0: padding; 1: unknown; 2...: vocabulary
    return ({v: k + 2 for k, v in enumerate(vocab_rev)},
            ['<PAD>', '<UNK>'] + vocab_rev, origin_vocab_size)

def myround(x, base=5):
    return int(base * round(float(x) / base))

def heuristic_seq_len(series, heuristic_percent=0.9):
    series_length = map(lambda x: myround(len(x)), series)
    histogram = pd.DataFrame(Counter(series_length).items(),
        columns=['bin', 'count']).sort_values(by='bin')
    histogram['cumsum_percent'] = histogram['count'].cumsum() / \
        histogram['count'].sum()
    return int(histogram[histogram.cumsum_percent >= heuristic_percent]
        .bin.iloc[0])
