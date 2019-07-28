import os
import numpy as np
import torch.utils.data as data

class CodeSummarizerDataset(data.Dataset):
    def __init__(self, data_path, task):
        assert task in ['train', 'valid', 'test', 'total'], 'Invalid task option'
        self.task = task
        self.token = np.load(os.path.join(data_path, '%s.token.npy' % task))
        if task != 'total':
            self.desc = np.load(os.path.join(data_path, '%s.desc.npy' % task))
            assert self.token.shape[0] == self.desc.shape[0], 'Broken dataset'
        self.data_len = self.token.shape[0]

    def __getitem__(self, index):
        if self.task != 'total':
            return self.token[index].astype(np.int64), \
                    self.desc[index].astype(np.int64)
        return self.token[index].astype(np.int64)

    def __len__(self):
        return self.data_len
