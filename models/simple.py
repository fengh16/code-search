#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from common.model.simple_embedder import SimpleEmbedder

import pickle
words, embeddings = pickle.load(open('./data/pretrained/polyglot-en.pkl', 'rb'), encoding='iso-8859-1')
model = SimpleEmbedder(embeddings, 200)

print(model(
        torch.randint(10004, (5, 6)),
        torch.randint(10004, (5, 30)),
        torch.randint(10004, (5, 50)),
        torch.randint(10004, (5, 30))
    ).size())
