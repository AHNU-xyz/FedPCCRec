import copy
import itertools
from collections import defaultdict, Counter

import math
import pickle
import random
import sys
from statistics import mean
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from GNN import GNN2_caiyang, NGCF, UltraGCN, GNN2_caiyang_light
from utility.helper import *
from utility.batch_test import *
import warnings

from utility.parser import client_args

warnings.filterwarnings('ignore')
from time import time
import FileProcess
import setSeed
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg




def TF_IDF_weight(result_dict,user_items,number_nb):
    user_items_weight = defaultdict(list)
   
    for user, items in user_items.items():
    
        interaction_counts = np.array([(1 / len(items)) * np.log((number_nb+1) / result_dict[item]) for item in items],
                                          dtype=float)

        sum_values = np.sum(interaction_counts)


        interaction_counts_normalized = interaction_counts / sum_values


        for i, item in enumerate(items):
            weight = interaction_counts_normalized[i]
            user_items_weight[user].append((item, weight))
    user_items_weight = dict(user_items_weight)
    return user_items_weight
def compute_distillation_loss(student_output, teacher_output, temperature=1.0):


    student_prob = torch.log_softmax(student_output / temperature, dim=1)
    teacher_prob = torch.softmax(teacher_output / temperature, dim=1)

    distillation_loss = torch.nn.functional.kl_div(student_prob, teacher_prob, reduction='batchmean') * (
                    temperature ** 2)
    return distillation_loss

def get_client_model_path(client_id, dataset, model_dir):

    folder_index = client_id // 1000
    folder_name = f"{model_dir}{dataset}/{folder_index * 1000}-{(folder_index + 1) * 1000 - 1}/"


    os.makedirs(folder_name, exist_ok=True)


    return f"{folder_name}client_{client_id}.pkl"

def load_client_model(model, client_id, dataset, device, model_dir):

    model_path = get_client_model_path(client_id, dataset, model_dir)


    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No saved model found for client {client_id}. Using initial model.")

def save_client_model(model, client_id, dataset, model_dir="global_client_models_"):

    model_path = get_client_model_path(client_id, dataset, model_dir)


    torch.save(model.state_dict(), model_path)





def select_users_by_performance(strong_users, weak_users, n_fed_client_each_round, strong_user_ratio):

    n_strong_users = int(n_fed_client_each_round * strong_user_ratio)
    n_weak_users = n_fed_client_each_round - n_strong_users


    random.shuffle(strong_users) 
    random.shuffle(weak_users)   

    selected_strong_users = strong_users[:min(n_strong_users, len(strong_users))]
    selected_weak_users = weak_users[:min(n_weak_users, len(weak_users))]


    idxs_users = selected_strong_users + selected_weak_users

    return idxs_users

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)
if __name__ == '__main__':
    setSeed.init_seed(42)

    args = parse_args()
    args.device = torch.device('cuda:0')
    args.dataset = '100k'
    args.pi = False 
    args.add_noise = False
    args.noise = 0.1 
    args.epsilon = 0.01 
    args.pi_number = 0.1 
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, args=args)
    data_generator.print_statistics()
    plain_adj, norm_adj, mean_adj, adj_mat = data_generator.get_adj_mat_new()
    algorithm_name = 'global_test_100k_onlybce'
    args.mode = 'fednebmask'
    args.epoch = 2000
    args.local_epoch = 5 
    args.layers = 2 
    args.embed_size = 64
    args.lr = 0.001
    args.alpha = 1

    args.attention = True 
    args.number_neb = 5 
    args.Distillation = False
    temperature = 1
    t1 = time()
    similar_users_dict = data_generator.optimized_find_all_similar_users(plain_adj, args.number_neb)
    t2 = time()
    print("find_neb_time:", t2 - t1)
    t3 = time()
   
    user_performance = {}
    
    probability_1 = 0  
    probability_0 = 1-probability_1 
    select_strong_ratio = 0 
    
    for user_id in range(data_generator.n_users):
        performance_flag = random.choices([0, 1], weights=[probability_0, probability_1], k=1)[0]
        user_performance[user_id] = performance_flag
    
    strong_users = []
    weak_users = []
    for user_id, flag in user_performance.items():
        if flag == 1:
            strong_users.append(user_id)
        else:
            weak_users.append(user_id)
    
    count_0 = sum(1 for flag in user_performance.values() if flag == 0)
    count_1 = sum(1 for flag in user_performance.values() if flag == 1)
   
    print("User Performance:", user_performance)
    print(f"Number of 0s: {count_0}")
    print(f"Number of 1s: {count_1}")
    t4 = time()
    
  
    global_model = GNN2_caiyang(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 plain_adj,
                 args,
                 ).to(args.device)
   
    args_local = client_args()
    args_local.embed_size = 64
    args_local.layers = 2
    args_local.device = args.device
    args_local.attention = args.attention
    args_local.mode = args.mode
    args_local.number_neb = args.number_neb
    args_local.local_epoch = 5

    local_model = GNN2_caiyang_light(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 plain_adj,
                 args_local,
                 ).to(args.device)


    t0 = time()
    n_fed_client_each_round = 128

    n_client = data_generator.n_users
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(global_model.parameters(), lr=args.lr)
    loss_loger, pre_loger10,pre_loger20, rec_loger10,rec_loger20, \
        ndcg_loger10,ndcg_loger20, hit_loger10,hit_loger20, = [],[],[],[],[],[],[],[],[]
    best_hr_10, best_hr_20, best_ndcg_10, best_ndcg_20 = 0, 0, 0, 0
    training_time = 0.0,

    mask = torch.zeros(size=norm_adj.shape, device=args.device)


    begin_time = time()
    #client_model = {}
    norm_adj = _convert_sp_mat_to_sp_tensor(norm_adj).to(args.device).to_dense()
    plain_adj = _convert_sp_mat_to_sp_tensor(plain_adj).to(args.device).to_dense()

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        model_para_list = []
       
        user_ini_state = copy.deepcopy(global_model.state_dict())['embedding_dict.user_emb']
        user_emb_list = {}
       
        idxs_users = random.sample(range(0, n_client), n_fed_client_each_round)
        
        #idxs_users = select_users_by_performance(strong_users, weak_users, n_fed_client_each_round, select_strong_ratio)

        for idx in (idxs_users):
            
            model_ini = copy.deepcopy(global_model.state_dict())
            global_model_copy = copy.deepcopy(global_model)
            model = global_model
            local_epoch  = args.local_epoch
            if user_performance[idx] == 0:
                model = local_model
                local_epoch = args_local.local_epoch
               
                model.embedding_dict['user_emb'].data.copy_(global_model.embedding_dict['user_emb'])
                model.embedding_dict['item_emb'].data.copy_(global_model.embedding_dict['item_emb'])


           
            if args.attention and user_performance[idx] == 1:
                users = np.concatenate([[idx], similar_users_dict[idx]]) 
                pos_items_a = list(itertools.chain.from_iterable(data_generator.train_items[user] for user in users if user in data_generator.train_items)) 
                
                counter_dict = Counter(pos_items_a)
               
                result_dict = dict(counter_dict)
               
                user_items = {id: data_generator.train_items[id] for id in users if id in data_generator.train_items}
                user_items_weight = TF_IDF_weight(result_dict, user_items,args.number_neb)
               
                for user, items_weights in user_items_weight.items():
                    for item, weight in items_weights:
                        norm_adj[user, data_generator.n_users + item] = plain_adj[user, data_generator.n_users + item] * weight

           

            for i in range(local_epoch):

                
                users, pos_items, neg_items = data_generator.sample_local(idx, similar_users_dict)
                if args.Distillation:
                    with torch.no_grad():
                        teacher_output = global_model_copy.generate_soft_labels(users, pos_items, neg_items, temperature)
                if args.mode == 'fedneb':
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                                         pos_items,
                                                                                         neg_items,
                                                                                         0,
                                                                                         norm_adj
                                                                                         )
                else:
                    
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, mask = model(users,
                                                                                pos_items,
                                                                                neg_items,
                                                                                mask,
                                                                                norm_adj
                                                                                )
                
                batch_loss, batch_mf_loss, batch_emb_loss = model.create_cross_entropy_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
                if args.Distillation:
                  
                    student_output = model.forward_with_logits(users, pos_items, neg_items)
                    
                    distillation_loss = compute_distillation_loss(student_output, teacher_output, temperature)
                optimizer.zero_grad()
              
                total_loss = batch_loss
                if args.Distillation:
                    total_loss = batch_loss + args.alpha * distillation_loss
                total_loss.backward()


                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            
            # t13 = time()
            # model_aft= train_decoder_with_local_data(local_model, local_model.emb_size, global_model.emb_size, similar_users_dict, idx, epochs=20, lr=0.001)
            # t14 = time()
           


           
            # light_model.embedding_dict['user_emb'].data.copy_(model.embedding_dict['user_emb'])
            # light_model.embedding_dict['item_emb'].data.copy_(model.embedding_dict['item_emb'])
          
            # save_client_model(light_model, idx, args.dataset)

            model_aft = copy.deepcopy(model.state_dict())
           
            user_emb_list[idx] = model_aft['embedding_dict.user_emb'][idx]

            model_para_list += [model_aft]
            # if user_performance[idx] == 0:
            #     model_weak_para_list.append(model_aft)
            # else:
            #     model_strong_para_list.append(model_aft)
            

            
            global_model.load_state_dict(model_ini)

        w_ = FedAvg(model_para_list)
        #w_ = federated_averaging(model_strong_para_list,model_weak_para_list)
        #w_ = map_embeddings_linear(w_, local_model.emb_size, global_model.emb_size)
        #w_ = map_embeddings_random_projection(w_, local_model.emb_size, global_model.emb_size)
        #w_ = train_decoder_and_map_embeddings(w_, local_model.emb_size, global_model.emb_size, global_model)
        for j in user_emb_list:
            user_ini_state[j] = user_emb_list[j]
        w_['embedding_dict.user_emb'] = copy.deepcopy(user_ini_state)
        
        # partial_load_global_model(global_model, w_)
        global_model.load_state_dict(w_)

        loss_loger.append(loss.item())





        if (epoch+1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        # ret = test(model, users_to_test, drop_flag=False)
        hr10, hr20, ndcg10, ndcg20 = test_model(global_model, data_generator)
        # hr10,hr20,ndcg10,ndcg20 = client_model_test(model,data_generator,client_model)
        t3 = time()

        hit_loger10.append(hr10)
        hit_loger20.append(hr20)
        ndcg_loger10.append(ndcg10)
        ndcg_loger20.append(ndcg20)

        if hr10 > best_hr_10 or ndcg10 > best_ndcg_10 or hr20 > best_hr_20 or ndcg20 > best_ndcg_20:
            best_hr_10 = max(best_hr_10, hr10)
            best_ndcg_10 = max(best_ndcg_10, ndcg10)
            best_hr_20 = max(best_hr_20, hr20)
            best_ndcg_20 = max(best_ndcg_20, ndcg20)

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],  ' \
                    'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, hr10,hr20,ndcg10,ndcg20)
            print(perf_str)



        if hr10 == best_hr_10 and args.save_flag == 1:
            torch.save(global_model.state_dict(), args.weights_path + str(args.dataset)+'/' + str(algorithm_name) +'.pkl')
            print('save the weights in path: ', args.weights_path + str(args.dataset)+'/' + str(algorithm_name) + '.pkl')



    best_hr_10 = max(hit_loger10)
    idx = hit_loger10.index(best_hr_10)

    final_perf = "Best Iter=[%d]@[%.1f]\thit=[%.5f %.5f], ndcg=[%.5f %.5f]" % \
                 (idx, time() - t0, hit_loger10[idx], hit_loger20[idx], ndcg_loger10[idx], ndcg_loger20[idx])

    print(final_perf)


    save_dict = data_generator.save_dict()
    save_dict['algorithm_name'] = algorithm_name
    save_dict['model'] = args.mode
    save_dict['HR@10'] = hit_loger10[idx]
    save_dict['HR@20'] = hit_loger20[idx]
    save_dict['NDCG@10'] = ndcg_loger10[idx]
    save_dict['NDCG@20'] = ndcg_loger20[idx]
    save_dict['n_fed_client_each_round'] = n_fed_client_each_round
    save_dict['GCN_layer'] = args.layers
    save_dict['lr'] = args.lr
    save_dict['epoch'] = args.epoch
    save_dict['dataset'] = args.dataset
    save_dict['HR@10_list'] = hit_loger10
    save_dict['HR@20_list'] = hit_loger20
    save_dict['NDCG@10_list'] = ndcg_loger10
    save_dict['NDCG@20_list'] = ndcg_loger20
    save_dict['loss'] = loss_loger
    save_dict['n_neb'] = args.number_neb
    save_dict['embedding_size'] = args.embed_size
    save_dict['noise'] = args.noise
    save_dict['local_epoch'] = args.local_epoch
    save_dict['epsilon'] = args.epsilon
    save_dict['pi_number'] = args.pi_number
    FileProcess.add_row_test(save_dict)

