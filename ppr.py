import numpy as np
import torch
from tqdm import tqdm
from load_data import DataLoader
import time
from utils import *


def get_ppr(loader, bs=128, N=20):
    
    tkg = torch.LongTensor(loader.tKG).cuda()
    len_tkg = len(tkg)
    uni, count = torch.unique(tkg[:,0], dim=0, return_inverse=False, return_counts=True)

    id_c = torch.cat((torch.arange(loader.n_nodes).view(1,-1),torch.arange(loader.n_nodes).view(1,-1)), dim=0).cuda()
    val_c = 1.0 / count
    cnt = torch.sparse_coo_tensor(id_c, val_c, (loader.n_nodes,loader.n_nodes)).cuda()

    index = torch.cat((tkg[:,0].view(1,-1),tkg[:,2].view(1,-1)),dim=0).cuda()
    value = torch.ones(len_tkg).cuda()
    Mkg = torch.sparse_coo_tensor(index, value, (loader.n_nodes,loader.n_nodes)).cuda()

    M = torch.sparse.mm(Mkg, cnt).cuda()
    print('M ready')
    s_time = time.time()

    alpha = 0.85
    beta = 0.8

    n_user = loader.n_users
    n_batch = n_user // bs + (n_user % bs > 0)

    final_rank = torch.zeros(loader.n_nodes, n_user)

    for i in tqdm(range(n_batch)):
        if i* bs + bs > n_user:
            tbs = n_user - i*bs
        else:
            tbs = bs
        u_list = torch.arange(i*bs, i*bs+tbs) 
        u_index = torch.cat((torch.LongTensor(u_list).view(1,-1), torch.arange(tbs).view(1,-1)), dim = 0).cuda()
        u_value = torch.ones(tbs).cuda()
        U = torch.sparse_coo_tensor(u_index, u_value, (loader.n_nodes,tbs)).cuda()
 
        for j in range(tbs):
            
            p_list = loader.known_user_set[u_list[j].item()]
         
            if j == 0:
                p_index = torch.cat((torch.arange(loader.n_nodes).view(1,-1), torch.zeros(1,loader.n_nodes)), dim = 0).cuda()
                p_value = torch.zeros(1,loader.n_nodes).cuda()
                p_value = p_value + (1 - beta) / (loader.n_nodes - len(p_list))
                p_value[0,p_list] = beta / len(p_list)  if len(p_list)!= 0 else 0
            else:
                p_id = torch.cat((torch.arange(loader.n_nodes).view(1,-1), j*torch.ones(1,loader.n_nodes)), dim = 0).cuda()
                p_index = torch.cat((p_index,p_id), dim = 1).cuda()
                p_val = torch.zeros(1,loader.n_nodes).cuda()
                p_val = p_val + (1 - beta) / (loader.n_nodes - len(p_list))
                p_val[0,p_list] = beta / len(p_list)    if len(p_list)!= 0 else 0
                p_value = torch.cat((p_value, p_val), dim = 1).cuda()
        p_value = p_value.squeeze(0)
        P = torch.sparse_coo_tensor(p_index, p_value, (loader.n_nodes,tbs)).coalesce().cuda()

        err = []
        rank = U 
        for r in range(N):
            old_rank = rank
            rank = (1 - alpha) * P + alpha * torch.sparse.mm(M, rank) 
            error = rank - old_rank
            error = error._to_dense()
            en2 = torch.norm(error).item()
            err.append(en2)

        final_rank[:,i*bs: i*bs+tbs] = rank._to_dense().cpu()
    
    final_rank = final_rank.T
    print('ppr done. time:', time.time() - s_time)
    
    return final_rank

