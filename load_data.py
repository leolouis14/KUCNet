import os
import random
import torch
from scipy.sparse import csr_matrix
import numpy as np
import time
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir
      
        if  task_dir == 'data/Dis_5fold_user/' or task_dir == 'data/Dis_5fold_item/' or  task_dir == 'data/new_last-fm/'or  task_dir == 'data/new_amazon-book/'or  task_dir == 'data/new_alibaba-fashion/':
            self.all_cf = self.read_cf(self.task_dir + 'train_1.txt')  # all_cf  (np.array)
            self.test_cf = self.read_cf(self.task_dir + 'test_1.txt')
        else:
            self.all_cf = self.read_cf(self.task_dir + 'train.txt') 
            self.test_cf = self.read_cf(self.task_dir + 'test.txt')

        self.n_users = max(max(self.all_cf[:,0]),max(self.test_cf[:,0])) + 1    
        self.n_items = max(max(self.all_cf[:,1]),max(self.test_cf[:,1])) + 1
        self.known_user_set = self.cf_to_set(self.all_cf) 
        self.test_user_set = self.cf_to_set(self.test_cf)
       
        n_all = self.all_cf.shape[0]
        rand_idx = np.random.permutation(n_all)
        self.all_cf = self.all_cf[rand_idx]

        self.triple = self.read_triples('kg.txt')   

        self.arraytriple = np.asarray(self.triple)   
        self.n_ent = max(max(self.arraytriple[:, 0]), max(self.arraytriple[:, 2])) + 1  
        self.n_nodes = self.n_ent + self.n_users    # user + entity            
        self.n_rel = max(self.arraytriple[:,1]) + 1  

        if task_dir == 'data/Dis_ind_item/' or  task_dir == 'data/Dis_5fold_item/'or  task_dir == 'data/ind_last-fm/'or  task_dir == 'data/ind_amazon-book/'or  task_dir == 'data/ind_alibaba-fashion/':
            self.item_set = self.cf_to_item_set(self.all_cf)
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
        else:
            self.facts_cf = self.all_cf[0:n_all*6//7]    
            self.train_cf = self.all_cf[n_all*6//7:]

        self.fact_triple = self.cf_to_triple(self.facts_cf)      
        self.train_triple = self.cf_to_triple(self.train_cf)  
        self.test_triple = self.cf_to_triple(self.test_cf)
        
        # add inverse
        self.d_triple  = self.double_triple(self.triple)
        
        # add interact and user into triplets
        self.fact_data, self.known_data = self.interact_triple(self.d_triple)   

        # use user-kg 
        if  task_dir == 'data/Dis_5fold_user/' or task_dir == 'data/Dis_5fold_item/' :
            self.readukg = 1
            self.ukg = self.read_user_kg()  
            self.n_rel += 1
            self.fact_data += self.ukg 
            self.known_data += self.ukg
            print('load user kg')
        else:
            self.readukg = 0

        self.load_graph(self.fact_data)  
        self.load_test_graph(self.known_data)

        self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
        self.test_q,  self.test_a  = self.load_query(self.test_triple)
    
        self.n_train = len(self.train_q)
        self.n_test = len(self.test_q)

        print('n_facts:',len(self.facts_cf), 'n_test_cf:', len(self.test_cf), 'n_train:', self.n_train,  'n_test:', self.n_test)
        print('users:', self.n_users,'items:', self.n_items, 'other entities:', self.n_ent - self.n_items )        

    def read_cf(self, file_name):  
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for al in lines:
            tmps = al.strip()
            inters = [int(i) for i in tmps.split(" ")]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

        return np.array(inter_mat)   #[[u1,i1],[u1,i2],……,[un,in]]   
    
    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = int(h), int(r), int(t)
                        
                triples.append([h,r,t])
                       
        return triples                                  

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 

        return triples + new_triples 
    
    def interact_triple(self, triples):
        # consider  additional relations  'interact' and 'in-interact'.
        copy_tri = triples.copy()
        for id, [h, r, t] in enumerate(copy_tri):
            copy_tri[id] = [h + self.n_users, r + 2, t + self.n_users] 
        
        fact_user_triple = []
        
        for uitriple in self.fact_triple:
            u , i = uitriple[0], uitriple[2]
            
            fact_user_triple.append([u, 0, i])  
            fact_user_triple.append([i, 1, u])    

        train_user_triple = []
        for uitriple in self.train_triple:
            u , i = uitriple[0], uitriple[2]
            
            train_user_triple.append([u, 0, i])  
            train_user_triple.append([i, 1, u])  
        return copy_tri + fact_user_triple,   copy_tri + fact_user_triple + train_user_triple
    
    def cf_to_triple(self, cf):
        cf = list(cf)
        newtriple = []
        for [u,i] in cf :
            if u >= self.n_users:
                continue
            i = i + self.n_users
            newtriple.append([u, 0, i ])
        return newtriple

    def cf_to_set(self, cf):
        cf = list(cf)
        user_set = defaultdict(list)
        for [u, i] in cf :
            if u >= self.n_users:
                print("unexpected users")
                continue
            user_set[u].append(i + self.n_users)
        return user_set
    
    def read_user_kg(self):
        ukg = []
        with open(os.path.join(self.task_dir, 'ukg.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = int(h), int(r), int(t)
                if h >= self.n_users or t >= self.n_users:
                    continue
                ukg.append([h, 2*self.n_rel + 2, t])
                ukg.append([t, 2*self.n_rel + 3, h])
        return ukg
        

    def cf_to_item_set(self, cf):  # item_set[i] = [u1, u3, u4……]
        cf = list(cf)
        item_set = defaultdict(list)
        for [u, i] in cf :
            if u >= self.n_users:
                print("unexpected users")
                continue
            item_set[i].append(u)
        return item_set

    def check_item_inkg(self, triple):    # return d(item) in kg
        it_inkg = np.zeros((self.n_items,1))
        for h,r,t in triple:
            if h < self.n_items:
                it_inkg[h] += 1
            if t < self.n_items:
                it_inkg[t] += 1
        return it_inkg
    
    def generate_inductive_train(self, cf):
        fcf = cf.tolist()
        n_train = 0
        train_cf = []
        ind_item = []
        while(n_train < len(cf) / 8):
            item = random.randint(0, self.n_items-1)
            if item in ind_item:
                continue
            for u in self.item_set[item]:
                train_cf.append([u,item])
                fcf.remove([u,item])      # as fact_cf
            ind_item.append(item)
            n_train += len(self.item_set[item])

        return np.array(fcf), np.array(train_cf)

    def load_graph(self, triples):      
        idd = np.concatenate([np.expand_dims(np.arange(self.n_nodes),1), (2*self.n_rel+2)*np.ones((self.n_nodes, 1)), np.expand_dims(np.arange(self.n_nodes),1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
 
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_nodes))


    def load_test_graph(self, triples):  
        idd = np.concatenate([np.expand_dims(np.arange(self.n_nodes),1), (2*self.n_rel+2)*np.ones((self.n_nodes, 1)), np.expand_dims(np.arange(self.n_nodes),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
    
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_nodes))

    def load_train_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        pos_items = defaultdict(lambda:list())
        neg_items = defaultdict(list)
        
        for trip in triples:
            h, r, t = trip            
            pos_items[(h,r)].append(t)
            while True:
                neg_item = np.random.randint(low=self.n_users, high=self.n_users + self.n_items, size=1)[0]  #重要修改！low=n_users
                if neg_item not in self.known_user_set[h]:
                    break
            neg_items[(h,r)].append(neg_item)
        
        queries = []
        answers = []
        wrongs = []
        for key in pos_items:
            queries.append(key)
            answers.append(np.array(pos_items[key]))
            wrongs.append(np.array(neg_items[key]))
        
        return queries, answers, wrongs


    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub
        
        # nodes: n(node) x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_nodes, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2 + 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'): 
        if data=='train':                                   
            query, answer, wrongs = np.array(self.train_q), self.train_a, self.train_w
        if data=='test':
            query, answer = np.array(self.test_q), np.array(self.test_a)

        subs = []
        rels = []
        objs = []
        pos = []
        neg =[]
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
  
        if data =='train':
            pos = answer[batch_idx[0]:batch_idx[-1]+1]
            neg = wrongs[batch_idx[0]:batch_idx[-1]+1]
            return subs, rels, pos, neg
        else :
            objs = np.zeros((len(batch_idx), self.n_nodes))
            for i in range(len(batch_idx)):
                objs[i][answer[batch_idx[i]]] = 1
            return subs, rels, objs


    def shuffle_train(self,):  
        
        if self.task_dir == 'data/Dis_5fold_item/' : 
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
            self.fact_triple = self.cf_to_triple(self.facts_cf)      
            self.train_triple = self.cf_to_triple(self.train_cf) 
            
            self.fact_data,_ = self.interact_triple(self.d_triple)
            if self.readukg == 1:
                self.fact_data += self.ukg 
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        elif self.task_dir == 'data/ind_last-fm/'or  self.task_dir == 'data/ind_amazon-book/'or  self.task_dir == 'data/ind_alibaba-fashion/':
            self.train_triple = np.array(self.train_triple)
            n_all = len(self.train_triple)
            rand_idx = np.random.permutation(n_all)
            self.train_triple = self.train_triple[rand_idx].tolist()
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        else:
            fact_triple = np.array(self.fact_triple)
            train_triple = np.array(self.train_triple)
            all_ui_triple = np.concatenate([fact_triple, train_triple], axis=0)
            n_all = len(all_ui_triple)
            rand_idx = np.random.permutation(n_all)
            all_ui_triple = all_ui_triple[rand_idx]

            self.fact_triple = all_ui_triple[:n_all*6//7].tolist()
            self.train_triple = all_ui_triple[n_all*6//7:].tolist()
            
            self.fact_data,_ = self.interact_triple(self.d_triple)
            if self.readukg == 1:
                self.fact_data += self.ukg 
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)

