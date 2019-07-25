import torch
import torch.nn as nn
import torch.nn.functional as F

pool_choices = {
    'max': lambda *args, **kwargs: torch.max(*args, **kwargs).values,
    'mean': torch.mean,
    'sum': torch.sum
}

rnn_choices = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}

activations_choices = {
    'relu': F.relu,
    'tanh': torch.tanh
}
