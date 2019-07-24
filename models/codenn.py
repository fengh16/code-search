#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
import spacy

from common.model.joint_embedder import JointEmbedder
from common.dataset.code_search import CodeSearchDataset
from common.eval import ACC, MAP, MRR, NDCG
from common.data_parallel import MyDataParallel

def eval(model, data_loader, device, pool_size, K):
    accs, mrrs, maps, ndcgs = [], [], [], []
    for names, apis, tokens, descs, _ in tqdm(data_loader, desc='Valid'):
        names, apis, tokens, descs = [tensor.to(device) for tensor in
            (names, apis, tokens, descs)]
        code_repr = model.forward_code(names, apis, tokens)
        descs_repr = model.forward_desc(descs)
        for i in range(pool_size):
            desc_repr = descs_repr[i].expand(pool_size, -1)
            sims = F.cosine_similarity(code_repr, desc_repr).data.cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict = predict[:K]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data / np.linalg.norm(data, axis=1).reshape((data.shape[0], 1))
    return normalized_data

if __name__ == '__main__':
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
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids splited by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size')
    parser.add_argument('--embed_size', type=int, default=100, help='embedding size')
    parser.add_argument('--repr_size', type=int, default=100, help='representation size; '
                        'for bidirectional rnn, the real value will be doubled')
    parser.add_argument('--pool', choices=['max', 'mean', 'sum'], default='max',
                        help='pooling method to use')
    parser.add_argument('--rnn', choices=['lstm', 'gru', 'rnn'], default='lstm',
                        help='rnn and rnn variants to use')
    parser.add_argument('--bidirectional', choices=['true', 'false'], default='true',
                        help='whether to use bidirectional rnn')
    parser.add_argument('--activation', choices=['relu', 'tanh'], default='tanh',
                        help='activation function to use')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='margin to use in the loss function')
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
    args = parser.parse_args()
    assert args.dataset_path is not None or args.task in ['search'], \
        '%s task requires dataset' % args.task
    assert args.load > 0 or args.task in ['train'], \
        "it's nonsense to %s on an untained model" % args.task
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    model = JointEmbedder(args.vocab_size, args.embed_size, args.repr_size,
                          args.pool, args.rnn, args.bidirectional == 'true',
                          args.activation, args.margin)
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
    if args.task == 'train':
        dataset = CodeSearchDataset(args.dataset_path, 'train', args.name_len, args.api_len,
                                    args.token_len, args.desc_len)
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
                                                      args.token_len, args.desc_len)
                    valid_data_loader = DataLoader(dataset=valid_dataset,
                                                   batch_size=args.eval_pool_size,
                                                   shuffle=True, drop_last=True)
                model.eval()
                acc, mrr, map, ndcg = eval(model, valid_data_loader,
                    device, args.eval_pool_size, args.eval_k)
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
                                    args.api_len, args.token_len, args.desc_len)
        data_loader = DataLoader(dataset=dataset, batch_size=args.eval_pool_size,
                                 shuffle=True, drop_last=True)
        print('ACC=%f, MRR=%f, MAP=%f, nDCG=%f' % eval(
            model, data_loader, device, args.eval_pool_size, args.eval_k))
    elif args.task == 'repr':
        model.eval()
        dataset = CodeSearchDataset(args.dataset_path, 'use', args.name_len,
                                    args.api_len, args.token_len)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
        vecs = None
        for data in tqdm(data_loader, desc='Repr'):
            data = [x.to(device) for x in data]
            reprs = model.forward_code(*data).data.cpu().numpy()
            vecs = reprs if vecs is None else np.concatenate((vecs, reprs), 0)
        vecs = normalize(vecs)
        print('saving codes to use.codes.pkl')
        with open(os.path.join(args.model_path, 'use.codes.pkl'), 'wb') as f:
            pickle.dump(vecs, f)
    elif args.task == 'search':
        model.eval()
        with open(os.path.join(args.model_path, 'use.codes.pkl'), 'rb') as f:
            reprs = pickle.load(f)
        code = pd.read_csv(os.path.join(args.dataset_path, 'use.codemap.csv'))
        assert reprs.shape[0] == code.shape[0], 'Broken data'
        with open(os.path.join(args.dataset_path, 'vocab.desc.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        nlp = spacy.load('en_core_web_lg')
        while True:
            try:
                query = input('> ')
            except:
                break
            words = [vocab.get(token.lemma_, 1) for token in nlp(query)
                        if token.is_alpha and not token.is_stop]
            if len(words) == 0:
                continue
            desc = torch.from_numpy(np.expand_dims(np.array(words), axis=0))
            desc = desc.to(device)
            desc_repr = model.forward_desc(desc).data.cpu().numpy()
            sim = np.dot(reprs, desc_repr.transpose()).squeeze(axis=1)
            idx = np.argsort(sim)[:args.search_top_n]
            for i in idx:
                record = code.loc[i]
                print('==== %s:%d:%d ====' % (record['file'], record['start'], record['end']))
                with open(record['file']) as f:
                    print(''.join(f.readlines()[record['start'] - 1:record['end']]))
