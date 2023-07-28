import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import torch

def cal_bpr_loss(n_users, pos, neg, scores):
 
    n = scores.shape[0]
    #pos_score, neg_score = [], []
    loss = 0
    for i in range(n):
        pos_score = scores[i][pos[i]-n_users]
        neg_score = scores[i][neg[i]-n_users]
        u_loss = -1*torch.sum(torch.nn.LogSigmoid()(pos_score - neg_score))
        loss += u_loss
    
    #loss = loss/n

    return loss

def ndcg_k(r, k, len_pos_test):

    if len_pos_test > k :
        standard = [1.0] * k
    else:
        standard = [1.0]*len_pos_test + [0.0]*(k - len_pos_test)
    dcg_max = dcg_k(standard, k)
    
    return dcg_k(r, k) / dcg_max

def dcg_k(r, k):

    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def cal_bceloss(users, pos, neg, scores):

    n = scores.shape[0]
    
    loss = 0
    for i in range(n):
        pos_score = scores[i][pos[i]]
        neg_score = scores[i][neg[i]]
        u_loss = -1*torch.sum(torch.nn.LogSigmoid()(pos_score) + torch.log(1 - torch.sigmoid(neg_score)))  
        loss += u_loss
    
    #loss = loss/n

    return loss


def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i
