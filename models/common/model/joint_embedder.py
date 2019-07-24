import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer.encoder import BOWEncoder, SeqEncoder

class JointEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_size, repr_size,
                 pool='max', rnn='lstm', bidirectional=True,
                 activation='tanh', margin=0.05):
        super(JointEmbedder, self).__init__()
        self.name_encoder = SeqEncoder(vocab_size, embed_size, repr_size,
                                       rnn=rnn, bidirectional=bidirectional,
                                       pool=pool, activation=activation)
        self.api_encoder = SeqEncoder(vocab_size, embed_size, repr_size,
                                      rnn=rnn, bidirectional=bidirectional,
                                      pool=pool, activation=activation)
        self.token_encoder = BOWEncoder(vocab_size, embed_size,
                                        pool=pool, activation=activation)
        self.desc_encoder = SeqEncoder(vocab_size, embed_size, repr_size,
                                       rnn=rnn, bidirectional=bidirectional,
                                       pool=pool, activation='tanh')
        if bidirectional:
            self.fuse = nn.Linear(embed_size + 4 * repr_size, 2 * repr_size)
        else:
            self.fuse = nn.Linear(embed_size + 2 * repr_size, repr_size)
        self.margin = margin

    def forward_code(self, name, api, token):
        name_repr = self.name_encoder(name)
        api_repr = self.api_encoder(api)
        token_repr = self.token_encoder(token)
        code_repr = self.fuse(torch.cat((name_repr, api_repr, token_repr), 1))
        return torch.tanh(code_repr)

    def forward_desc(self, desc):
        return self.desc_encoder(desc)

    def forward(self, name, api, token, desc_good, desc_bad):
        code_repr = self.forward_code(name, api, token)
        good_sim = F.cosine_similarity(code_repr, self.forward_desc(desc_good))
        bad_sim = F.cosine_similarity(code_repr, self.forward_desc(desc_bad))
        return (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
