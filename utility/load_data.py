import random
import sys
from itertools import combinations

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import utility.metrics
import torch
from sklearn.metrics.pairwise import cosine_similarity
import math

class Data(object):
    def __init__(self, path, batch_size,args):
        self.path = path
        self.batch_size = batch_size
        self.args = args
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')

                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])

                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    # print(items)
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

      
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}

        if args.add_noise:
            
            epsilon = args.epsilon
            p = args.noise
            self.k = int(self.n_items*p)
           
            self.probabilities = []
            for num_flips in range(1, self.k + 1):
                probability = np.exp(epsilon * (-num_flips) /2)
                self.probabilities.append(probability)
                #print(probability)
            self.probabilities /= np.sum(self.probabilities)
            #print("probabilities:", self.probabilities)

            # sampled_num_flips = np.random.choice(range(1, k + 1), p=probabilities)
            # print("sampled_num_flips:", sampled_num_flips)
            
            # indices = range(self.n_items)
            # self.comb = np.random.choice(indices, sampled_num_flips, replace=False)
            # print("sampled_num_flips:", self.comb)
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    if args.pi:
                    
                        p = args.pi_number
                        k = int(len(train_items)*p)
                        wei_list =[]
                        while(k > 0):
                            wei_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                            if wei_id not in train_items and wei_id not in wei_list:
                                wei_list.append(wei_id)
                                self.R[uid, wei_id] = 1.
                                k = k - 1
                    if args.add_noise:
                        sampled_num_flips = np.random.choice(range(1, self.k + 1), p = self.probabilities)
                        #print("sampled_num_flips:", sampled_num_flips)
                     
                        indices = range(self.n_items)
                        comb = np.random.choice(indices, sampled_num_flips, replace=False)
                        for i in comb:
                            self.R[uid, i] = 1 - self.R[uid, i]

                    
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            att_adj_mat = sp.load_npz(self.path + '/s_att_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat, att_adj_mat= self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            sp.save_npz(self.path + '/s_att_adj_mat.npz', att_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat, att_adj_mat

    def get_adj_mat_new(self):
        t1 = time()
        adj_mat, norm_adj_mat, mean_adj_mat, att_adj_mat = self.create_adj_mat()
        print('adj matrix create successfully!', time() - t1)
        return adj_mat, norm_adj_mat, mean_adj_mat, att_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # if self.args.add_noise:
        #     noise_mat = np.random.laplace(0, self.args.noise, self.R.shape)
        #     R_dense = self.R.toarray() 
        #     R_dense += noise_mat
        #     
        #     R_dense[R_dense >= 0.5] = 1
        #     R_dense[R_dense < 0.5] = 0
        #    
        #     self.R = sp.dok_matrix(R_dense)
        R = self.R.tolil()
        '''
            [     R
             R.T   ]
        '''
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        t2 = time()
        print('already create adjacency matrix', adj_mat.shape, 'time:',t2 - t1)


        def att_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            rowsum[self.n_users:] = 1
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            # half_D(-1)*A
            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def mean_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            # D(-1)*A
            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            # D(-1/2)*A*D(-/2)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        #norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

      
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), adj_mat


    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample_test(self, idx):
        return self.test_set[idx]

    def sample_test_nagative(self, idx):
        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items and neg_id not in self.test_set[u]:
                    neg_items.append(neg_id)
            return neg_items

        neg_items_test = sample_neg_items_for_u(idx, 99)

        return self.test_set[idx], neg_items_test


    def sample_neb_1_n(self, idx,similar_users_dict):
        length = 1/5
        users = [idx] * len(self.train_items[idx])
        for id_ne in similar_users_dict[idx]:
            list_ne = [id_ne] * int(len(self.train_items[id_ne])*length)
            users.extend(list_ne)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items


    def sample_neb_all_1_n(self, idx,similar_users_dict):
        length = 1/5
        users = [idx] * int(len(self.train_items[idx])*length)
        for id_ne in similar_users_dict[idx]:
            list_ne = [id_ne] * int(len(self.train_items[id_ne])*length)
            users.extend(list_ne)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_neb_all(self, idx,similar_users_dict,number_neb):
        '''
        :users :
        :return:
        '''
        users = [idx] * len(self.train_items[idx])
        for id_ne in similar_users_dict[idx]:
            list_ne = [id_ne] * len(self.train_items[id_ne])
            users.extend(list_ne)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)


        pos_item = self.train_items[idx]
        neg_item = []
 
        for id_ne in similar_users_dict[idx]:
            pos = self.train_items[id_ne]
            pos_item= np.concatenate((pos_item, pos))

        for u in users:
            neg_item += sample_neg_items_for_u(u, 1)

        return users, pos_item, neg_item


    def sample_1_4(self, idx,similar_users_dict,number_neb):
        '''
        :users :
        :return:
        '''
        users = [idx] * len(self.train_items[idx])
        for id_ne in similar_users_dict[idx]:
            list_ne = [id_ne] * len(self.train_items[id_ne])
            users.extend(list_ne)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)


        pos_item = self.train_items[idx]
        neg_item = []

        for id_ne in similar_users_dict[idx]:
            pos = self.train_items[id_ne]
            pos_item= np.concatenate((pos_item, pos))

        for u in users:
            neg_item += sample_neg_items_for_u(u, 4)

        return users, pos_item, neg_item
    # just test
    def sample_local(self, idx,similar_users_dict):


        users = [idx]
        for id_ne in similar_users_dict[idx]:
            list_ne = [id_ne]
            users.extend(list_ne)

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 4)

        return users, pos_items, neg_items
    '''
    :return user[idx,idx,...,idx],neg_items=[],pos_items=[]
    '''
    def sample(self, idx):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        users = [idx] * len(self.train_items[idx])


        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    '''
    :return user[ ],pos_items[],neg_items[]
    '''
    def sample_central(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items


    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
 
    def save_dict(self):
        str_dict = {'algorithm_name': "", 'model':"",'noise':0,'HR@10':0,'HR@20':0,'NDCG@10':0,'NDCG@20':0,'n_fed_client_each_round':0,'GCN_layer':0,'embedding_size':0,'lr':0,'epoch':0,'dataset':"",'HR@10_list':[]
                ,'HR@20_list':[],'NDCG@10_list':[],'NDCG@20_list':[],'loss':[],'n_neb':0,'local_epoch':0,'epsilon':0,'pi_number':0
                    }
        return str_dict

    def find_similar_users(self,trainitem, user_id, top_n=5):
  
        

        target_user_items = set(trainitem[user_id])


        similarities = []
        for other_user_id, other_user_items in trainitem.items():
            if other_user_id != user_id:
                jaccard_sim = utility.metrics.Jaccard(target_user_items, set(other_user_items))
                similarities.append((other_user_id, jaccard_sim))

    
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in similarities[:top_n]]
 
    def find_all_similar_users(self,trainitem,number_neb):
        all_similar_users = {}
        for user_id in trainitem:
            similar_users = self.find_similar_users(trainitem, user_id,number_neb)
            all_similar_users[user_id] = similar_users
        return all_similar_users


    def optimized_find_all_similar_users(self,interaction_matrix, number_neb):
    
        user_item_matrix = interaction_matrix[:self.n_users, self.n_users:]

       
        similarity_matrix = cosine_similarity(user_item_matrix)

     
        all_similar_users = {}
        for i in range(self.n_users):
         
            similar_indices = np.argsort(similarity_matrix[i])[::-1][1:number_neb + 1]
            all_similar_users[i] = similar_indices.tolist()

        return all_similar_users

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
