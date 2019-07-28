#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import json
import datetime
import atexit
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter

from common.model.seq2seq import BaselineSeq2Seq, seq2seq_eval
from common.dataset.code_summarizer import CodeSummarizerDataset
from common.preprocessing import ktext_process_text
from common.data_parallel import MyDataParallel, NoDataParallel

if __name__ == '__main__':
    running = {
        'start': str(datetime.datetime.now()),
        'end': None,
        'argv': sys.argv,
        'parameters': {},
        'state': 'failed'
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train', 'valid', 'test', 'summarize'],
                        default='train', help="task to run; `train' for "
                        "training the dataset; `valid'/`test' for evaluating model "
                        "on corresponding dataset; `summarize' for summarize the "
                        "user input code")
    parser.add_argument('--dataset_path', help='path to the dataset', required=True)
    parser.add_argument('--model_path', help='path for saving models and codes',
                        required=True)
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids splited by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--embed_size', type=int, default=800, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='hidden state size')
    parser.add_argument('--epoch', type=int, default=200, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--valid_every_epoch', type=int, default=5,
                        help='run validation every numbers of epoch; 0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model every numbers of epoch; 0 for disabling')
    parser.add_argument('--comment', default='', help='comment for tensorboard')
    args = parser.parse_args()
    running['parameters'] = vars(args)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    def save_running_log():
        print('saving running log to running-log.json')
        running['end'] = str(datetime.datetime.now())
        filename = os.path.join(args.model_path, 'running-log.json')
        all_running = []
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                all_running = json.load(f)
        all_running.insert(0, running)
        with open(filename, 'w') as f:
            json.dump(all_running, f, indent=2)
    atexit.register(save_running_log)
    assert args.load > 0 or args.task in ['train'], \
        "it's nonsense to %s on an untained model" % args.task
    with open(os.path.join(args.dataset_path, 'word2code.token.pkl'), 'rb') as f:
        token_word2code = pickle.load(f)
    with open(os.path.join(args.dataset_path, 'code2word.token.pkl'), 'rb') as f:
        token_code2word = pickle.load(f)
    with open(os.path.join(args.dataset_path, 'word2code.desc.pkl'), 'rb') as f:
        desc_word2code = pickle.load(f)
    with open(os.path.join(args.dataset_path, 'code2word.desc.pkl'), 'rb') as f:
        desc_code2word = pickle.load(f)
    model = BaselineSeq2Seq(len(token_code2word), len(desc_code2word),
        args.embed_size, args.hidden_size)
    if args.gpu:
        model = MyDataParallel(model, device_ids=args.gpu)
    else:
        model = NoDataParallel(model)
    device = torch.device("cuda:%d" % args.gpu[0] if args.gpu else "cpu")
    optimizer_state_dict = None
    if args.load > 0:
        print('loading from epoch.%04d.pth' % args.load)
        model_state_dict, optimizer_state_dict = torch.load(
            os.path.join(args.model_path, 'epoch.%04d.pth' % args.load),
            map_location='cpu')
        model.load_state_dict(model_state_dict)
    model.to(device)
    running['state'] = 'interrupted'
    if args.task == 'train':
        dataset = CodeSummarizerDataset(args.dataset_path, 'train')
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 shuffle=True, drop_last=True)
        valid_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        criterion = nn.NLLLoss()
        writer = SummaryWriter(comment=args.comment)
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(data_loader, desc='Iter'), 1):
                data = [x.to(device) for x in data]
                token, desc = data
                encoder_input = token.transpose(0, 1)
                decoder_input = desc[:, :-1].transpose(0, 1)
                decoder_target = desc[:, 1:].transpose(0, 1)
                decoder_output = model(encoder_input, decoder_input)
                loss = criterion(decoder_output.transpose(1, 2), decoder_target)
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter % args.log_every_iter == 0:
                    tqdm.write('epoch:[%d/%d] iter:[%d/%d] Loss=%.5f' %
                               (epoch, args.epoch, iter, len(data_loader), np.mean(losses)))
                    losses = []
                step += 1
            if args.valid_every_epoch and epoch % args.valid_every_epoch == 0:
                if valid_data_loader is None:
                    valid_dataset = CodeSummarizerDataset(args.dataset_path, 'valid')
                    valid_data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
                model.eval()
                bleu = seq2seq_eval(model, device, valid_data_loader,
                                    desc_word2code['<SOS>'], desc_word2code['<EOS>'])
                tqdm.write('BLEU=%f' % bleu)
                writer.add_scalar('eval/bleu', bleu, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                    os.path.join(args.model_path, 'epoch.%04d.pth' % epoch))
