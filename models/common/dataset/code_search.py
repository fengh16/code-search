import os
import random
import numpy as np
import tables
import torch.utils.data as data

def pad_seq(seq, length):
    if seq.shape[0] < length:
        return np.append(seq, np.zeros((length - seq.shape[0],), dtype=seq.dtype))
    else:
        return seq[:length]


class CodeSearchDataset(data.Dataset):
    def __init__(self, data_path, task, name_len, api_len, token_len, desc_len=None):
        assert task in ['train', 'valid', 'test', 'use'], 'Invalid task option'
        self.task = task
        self.name_len, self.api_len, self.token_len, self.desc_len = \
            name_len, api_len, token_len, desc_len
        table_name = tables.open_file(os.path.join(data_path, '%s.methname.h5' % task))
        self.name = table_name.get_node('/phrases')
        self.name_idx = table_name.get_node('/indices')
        table_api = tables.open_file(os.path.join(data_path, '%s.apiseq.h5' % task))
        self.api = table_api.get_node('/phrases')
        self.api_idx = table_api.get_node('/indices')
        table_token = tables.open_file(os.path.join(data_path, '%s.tokens.h5' % task))
        self.token = table_token.get_node('/phrases')
        self.token_idx = table_token.get_node('/indices')
        if task != 'use':
            table_desc = tables.open_file(os.path.join(data_path, '%s.desc.h5' % task))
            self.desc = table_desc.get_node('/phrases')
            self.desc_idx = table_desc.get_node('/indices')

        assert self.name_idx.shape[0] == self.api_idx.shape[0], 'Broken dataset'
        assert self.name_idx.shape[0] == self.token_idx.shape[0], 'Broken dataset'
        if task != 'use':
            assert self.name_idx.shape[0]==self.desc_idx.shape[0], 'Broken dataset'
        self.data_len = self.name_idx.shape[0]

    def __getitem__(self, index):
        len, pos = self.name_idx[index]['length'], self.name_idx[index]['pos']
        name = self.name[pos:pos + len].astype('int64')
        name = pad_seq(name, self.name_len)
        len, pos = self.api_idx[index]['length'], self.api_idx[index]['pos']
        api = self.api[pos:pos+len].astype('int64')
        api = pad_seq(api, self.api_len)
        len, pos = self.token_idx[index]['length'], self.token_idx[index]['pos']
        token = self.token[pos:pos+len].astype('int64')
        token = pad_seq(token, self.token_len)
        if self.task != 'use':
            len, pos = self.desc_idx[index]['length'], self.desc_idx[index]['pos']
            desc_good = self.desc[pos:pos+len].astype('int64')
            desc_good = pad_seq(desc_good, self.desc_len)
            rand_index = random.randint(0, self.data_len - 2)
            if rand_index == index:
                rand_index = self.data_len - 1
            len, pos = self.desc_idx[rand_index]['length'], self.desc_idx[rand_index]['pos']
            desc_bad = self.desc[pos:pos+len].astype('int64')
            desc_bad = pad_seq(desc_bad, self.desc_len)
            return name, api, token, desc_good, desc_bad
        else:
            return name, api, token

    def __len__(self):
        return self.data_len
