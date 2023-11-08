import numpy as np
import torch

def cal_bpr_loss(n_users, pos, neg, scores):
 
    n = scores.shape[0]
    
    loss = 0
    for i in range(n):
        pos_score = scores[i][pos[i]-n_users]
        neg_score = scores[i][neg[i]-n_users]
        u_loss = -1*torch.sum(torch.nn.LogSigmoid()(pos_score - neg_score))
        loss += u_loss

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

    return loss

