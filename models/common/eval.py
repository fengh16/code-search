import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from .choices import similarity_choices

def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))

def MAP(real, predict):
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))

def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))

def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank  =  i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)

def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg

def eval(model, data_loader, device, pool_size, K, similarity='cos'):
    accs, mrrs, maps, ndcgs = [], [], [], []
    for names, apis, tokens, descs, _ in tqdm(data_loader, desc='Eval'):
        names, apis, tokens, descs = [tensor.to(device) for tensor in
            (names, apis, tokens, descs)]
        code_repr = model.forward_code(names, apis, tokens)
        descs_repr = model.forward_desc(descs)
        for i in range(pool_size):
            desc_repr = descs_repr[i].expand(pool_size, -1)
            sims = similarity_choices[similarity](code_repr,
                desc_repr).data.cpu().numpy()
            predict = np.argsort(sims)
            predict = predict[:K]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)
