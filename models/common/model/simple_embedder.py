import torch
import torch.nn as nn
import torch.nn.functional as F
from ..choices import pool_choices, activations_choices

class SimpleEmbedder(nn.Module):
    def __init__(self, embedding, hidden_size, pool='mean', activation='relu'):
        super(SimpleEmbedder, self).__init__()
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.embedding.requires_grad = False
        self.fc1 = nn.Linear(3 * embedding.shape[1], hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding.shape[1])
        assert pool in pool_choices.keys(), 'Invalid pool option'
        self.pool = pool_choices[pool]
        assert activation in activations_choices.keys(), 'Invalid activation option'
        self.activation = activations_choices[activation]

    def forward_code(self, api, seq, token):
        codes = [self.pool(self.embedding(input), dim=1)
            for input in [api, seq, token]]
        output = self.fc1(torch.cat(codes, dim=1))
        output = self.activation(F.dropout(output, 0.25, self.training))
        return self.fc2(output)

    def forward_desc(self, desc):
        return self.pool(self.embedding(desc), dim=1)

    def forward(self, api, seq, token, desc):
        code_repr = self.forward_code(api, seq, token)
        desc_repr = self.forward_desc(desc)
        return torch.mean((code_repr - desc_repr) ** 2, dim=1)
