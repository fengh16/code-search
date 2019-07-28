#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from common.model.seq2seq import BaselineSeq2Seq
from common.dataset.code_summarizer import CodeSummarizerDataset
from common.preprocessing import ktext_process_text

dataset_path = './data/dataset/py_github_newer-code_summarizer'

with open(os.path.join(dataset_path, 'word2code.desc.pkl'), 'rb') as f:
    desc_word2vec = pickle.load(f)
with open(os.path.join(dataset_path, 'code2word.desc.pkl'), 'rb') as f:
    desc_code2word = pickle.load(f)

def predict(model, input, max_len=20):
    state_hidden = model.forward_encoder(input.view(-1, 1))
    original_encoding = state_hidden
    input = torch.LongTensor([desc_word2vec['<SOS>']]).view(1, 1)
    output_sentence = []
    while len(output_sentence) < max_len:
        output, state_hidden = model.forward_decoder(input, state_hidden)
        # skip 0 (padding) and 1 (unknown)
        output_word = torch.argmax(output[:, :, 2:]).detach() + 2
        if desc_code2word[output_word.item()] == '<EOS>':
            break
        output_sentence.append(output_word)
        input = torch.LongTensor([output_word]).view(1, 1)
    return original_encoding, torch.stack(output_sentence)

model = BaselineSeq2Seq(20000, 14000, 800, 1000)
criterion = nn.NLLLoss()
dataset = CodeSummarizerDataset(dataset_path, 'train')
data_loader = DataLoader(dataset=dataset, batch_size=512,
                         shuffle=True, drop_last=True)
optimizer = Adam(model.parameters())

###
import random
###
for token, desc in data_loader:
    encoder_input = token.transpose(0, 1)
    decoder_input = desc[:, :-1].transpose(0, 1)
    decoder_target = desc[:, 1:].transpose(0, 1)
    decoder_output = model(encoder_input, decoder_input)
    loss = criterion(decoder_output.transpose(1, 2), decoder_target)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ###
    a, b = dataset[random.randint(0, len(dataset) - 1)]
    a = torch.from_numpy(a)
    print('origin: ' + ' '.join(map(lambda x: desc_code2word[x], b)))
    model.eval()
    _, b = predict(model, a)
    model.train()
    print('predicted: ' + ' '.join(map(lambda x: desc_code2word[x], b)))
    ###
