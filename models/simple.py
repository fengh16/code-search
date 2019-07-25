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
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
import spacy

from common.model.simple_embedder import SimpleEmbedder
from common.dataset.code_search import CodeSearchDataset
from common.eval import eval
from common.data_parallel import MyDataParallel

if __name__ == '__main__':
    running = {
        'start': str(datetime.datetime.now()),
        'end': None,
        'argv': sys.argv,
        'parameters': {},
        'state': 'failed'
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["train", "valid", "test", "repr", "search"],
                        default='train', help="task to run; `train' for training the "
                        "dataset; `valid'/`test' for evaluating model on corresponding "
                        "dataset; `repr' for converting whole dataset(`use') to code "
                        "`search' for searching in whole dataset, it require `repr' "
                        "to run first")
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--model_path', help='path for saving models and codes',
                        required=True)
    parser.add_argument('--word2vec', help='path to word2vec model pickle', required=True)
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids splited by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden unit size')
    parser.add_argument('--pool', choices=['max', 'mean', 'sum'], default='max',
                        help='pooling method to use')
    parser.add_argument('--activation', choices=['relu', 'tanh'], default='tanh',
                        help='activation function to use')
    parser.add_argument('--name_len', type=int, default=6, help='length of name sequence')
    parser.add_argument('--api_len', type=int, default=30, help='length of api sequence')
    parser.add_argument('--token_len', type=int, default=50, help='length of tokens')
    parser.add_argument('--desc_len', type=int, default=30,
                        help='length of description sequence')
    parser.add_argument('--epoch', type=int, default=100, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--eval_pool_size', type=int, default=1000,
                         help='pool size for evaluation')
    parser.add_argument('--eval_k', type=int, default=10, help='k for evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--valid_every_epoch', type=int, default=5,
                        help='run validation every numbers of epoch; 0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model every numbers of epoch; 0 for disabling')
    parser.add_argument('--search_top_n', type=int, default=5,
                        help='search top-n results for search task')
    parser.add_argument('--comment', default='', help='comment for tensorboard')
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
        assert args.dataset_path is not None or args.task in ['search'], \
        '%s task requires dataset' % args.task
    assert args.load > 0 or args.task in ['train'], \
        "it's nonsense to %s on an untained model" % args.task
    with open(args.word2vec, 'rb') as f:
        # <UNK>": 0, "<S>": 1, "</S>":2, "<PAD>": 3
        vocab_rev, embedding = pickle.load(f, encoding='iso-8859-1')
    model = SimpleEmbedder(embedding, args.hidden_size,
                           args.pool, args.activation)
    model = MyDataParallel(model, device_ids=args.gpu if args.gpu else None)
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
        dataset = CodeSearchDataset(args.dataset_path, 'train', args.name_len, args.api_len,
                                    args.token_len, args.desc_len, pad_idx=3)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 shuffle=True, drop_last=True)
        valid_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        writer = SummaryWriter(comment=args.comment)
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(data_loader, desc='Iter'), 1):
                data = data[:4]
                data = [x.to(device) for x in data]
                loss = model(*data).mean()
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
                    valid_dataset = CodeSearchDataset(args.dataset_path, 'valid',
                                                      args.name_len, args.api_len,
                                                      args.token_len, args.desc_len,
                                                      pad_idx=3)
                    valid_data_loader = DataLoader(dataset=valid_dataset,
                                                   batch_size=args.eval_pool_size,
                                                   shuffle=True, drop_last=True)
                model.eval()
                acc, mrr, map, ndcg = eval(model, valid_data_loader,
                    device, args.eval_pool_size, args.eval_k, 'l2')
                tqdm.write('ACC=%f, MRR=%f, MAP=%f, nDCG=%f' % (acc, mrr, map, ndcg))
                writer.add_scalar('eval/acc', acc, epoch)
                writer.add_scalar('eval/mrr', mrr, epoch)
                writer.add_scalar('eval/map', map, epoch)
                writer.add_scalar('eval/ndcg', ndcg, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                    os.path.join(args.model_path, 'epoch.%04d.pth' % epoch))
    elif args.task in ['valid', 'test']:
        model.eval()
        dataset = CodeSearchDataset(args.dataset_path, args.task, args.name_len,
                                    args.api_len, args.token_len, args.desc_len,
                                    pad_idx=3)
        data_loader = DataLoader(dataset=dataset, batch_size=args.eval_pool_size,
                                 shuffle=True, drop_last=True)
        print('ACC=%f, MRR=%f, MAP=%f, nDCG=%f' % eval(
            model, data_loader, device, args.eval_pool_size, args.eval_k, 'l2'))
    elif args.task == 'repr':
        model.eval()
        dataset = CodeSearchDataset(args.dataset_path, 'use', args.name_len,
                                    args.api_len, args.token_len, pad_idx=3)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
        vecs = None
        for data in tqdm(data_loader, desc='Repr'):
            data = [x.to(device) for x in data]
            reprs = model.forward_code(*data).data.cpu().numpy()
            vecs = reprs if vecs is None else np.concatenate((vecs, reprs), 0)
        print('saving codes to use.codes.pkl')
        with open(os.path.join(args.model_path, 'use.codes.pkl'), 'wb') as f:
            pickle.dump(vecs, f)
    elif args.task == 'search':
        model.eval()
        with open(os.path.join(args.model_path, 'use.codes.pkl'), 'rb') as f:
            reprs = pickle.load(f)
        code = pd.read_csv(os.path.join(args.dataset_path, 'use.codemap.csv'))
        assert reprs.shape[0] == code.shape[0], 'Broken data'
        vocab = {v: k for k, v in enumerate(vocab_rev)}
        nlp = spacy.load('en_core_web_lg')
        while True:
            try:
                query = input('> ')
            except:
                break
            words = [vocab.get(token.lemma_, 0) for token in nlp(query)
                        if token.is_alpha and not token.is_stop]
            if len(words) == 0:
                continue
            desc = torch.from_numpy(np.expand_dims(np.array(words), axis=0))
            desc = desc.to(device)
            desc_repr = model.forward_desc(desc).data.cpu().numpy()
            sim = np.linalg.norm(reprs - desc_repr, axis=1)
            idx = np.argsort(sim)[:args.search_top_n]
            for i in idx:
                record = code.loc[i]
                if 'code' in record.index:
                    print('========')
                    print(record['code'])
                else:
                    print('==== %s:%d:%d ====' % (record['file'],
                        record['start'], record['end']))
                    with open(record['file']) as f:
                        print(''.join(f.readlines()[record['start'] -
                            1:record['end']]).strip())
    running['state'] = 'succeeded'
