from random import random
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import heapq
from models import KUCNet_trans
from utils import *

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = KUCNet_trans(args, loader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model.to(self.device)

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_users = loader.n_users
        self.n_items = loader.n_items
        self.n_nodes = loader.n_nodes

        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.known_user_set = loader.known_user_set
        self.test_user_set = loader.test_user_set

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)

        self.smooth = 1e-5
        self.t_time = 0


    def train_batch(self,):  
        epoch_loss = 0
        i = 0
        
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        
        t_time = time.time()
        self.model.train()
        for i in range(n_batch):
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, pos, neg = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(subs, rels) 
           
            loss = cal_bpr_loss(self.n_users, pos, neg, scores)
            loss.backward()
            self.optimizer.step()

            if i % 600 == 0 :
                print('batch:',i, 'loss:', loss.item())
            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

        self.t_time += time.time() - t_time
        print('epoch_loss:',epoch_loss)
        print('start test')
        recall, ndcg, out_str = self.test_batch() 
        
        self.loader.shuffle_train()
        print(out_str)
        return recall, ndcg,  out_str

    def test_one_user(self, u, score, K = 20):
        try:
            training_items = self.known_user_set[u]
        except Exception:
            print('unexpected user!!!')
            training_items = []
        user_pos_test = self.test_user_set[u]

        all_items = set(range(self.n_users, self.n_users + self.n_items))

        test_items = list(all_items - set(training_items))

        item_score = {}
        for i in test_items:
            item_score[i] = score[i-self.n_users]
        K_item_score = heapq.nlargest(K, item_score, key=item_score.get)

        r = []
        for i in K_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        # ndcg
        ndcg = ndcg_k(r, K, len(user_pos_test))

        r = np.asarray(r)
        recall = np.sum(r) / len(user_pos_test)
        
        return recall, ndcg


    def test_batch(self, ):
        batch_size = self.n_tbatch

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        recall, ndcg = 0, 0
    
        for id in range(n_batch):
            start = id*batch_size
            end = min(n_data, (id+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()
        
            batch_recall, batch_ndcg = 0, 0
            for i in range(len(subs)):
                u , u_score = subs[i], scores[i]
                one_recall, one_ndcg, one_num, one_ratio, perf = self.test_one_user(u, u_score)
                batch_recall = batch_recall + one_recall
                batch_ndcg = batch_ndcg + one_ndcg
                
            recall = recall + batch_recall
            ndcg = ndcg + batch_ndcg
            batch_recall = batch_recall / len(subs)
            batch_ndcg = batch_ndcg / len(subs)
            if id % 500 == 0:
                print(id, 'batch recall:', batch_recall, 'batch ndcg:', batch_ndcg)

        recall = recall / n_data
        ndcg = ndcg / n_data

        i_time = time.time() - i_time

        out_str = '[TEST] recall:%.4f  ndcg:%.4f   [TIME] train:%.4f inference:%.4f\n'%( recall, ndcg, self.t_time, i_time)
        
        return recall, ndcg, out_str
    
