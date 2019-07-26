import torch.nn as nn

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class NoDataParallel(nn.Module):
    def __init__(self, module):
        super(NoDataParallel, self).__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
