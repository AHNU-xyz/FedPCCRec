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

# =============================================================================
# Federated Averaging Aggregation Function
# =============================================================================
def FedAvg(w):
    """
    Perform federated averaging to aggregate model parameters from multiple clients.
    
    Args:
        w: List of state dictionaries from different client models
        
    Returns:
        w_avg: Averaged state dictionary for the global model
    """
    # Initialize the averaged model with a deep copy of the first client's parameters
    w_avg = copy.deepcopy(w[0])
    # Iterate through each parameter key in the model
    for k in w_avg.keys():
        # Sum up the parameters from all clients
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        # Compute the average by dividing by the number of clients
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# =============================================================================
# TF-IDF Weight Calculation for Neighbor Interactions
# =============================================================================
def TF_IDF_weight(result_dict, user_items, number_nb):
    """
    Calculate TF-IDF weights for items in user interaction lists.
    This helps prioritize rare but informative items when aggregating neighbor information.
    
    Args:
        result_dict: Dictionary containing item frequency counts across all neighbors
        user_items: Dictionary mapping users to their interaction items
        number_nb: Number of neighbors considered (used for IDF smoothing)
        
    Returns:
        user_items_weight: Dictionary mapping users to list of (item, weight) tuples
    """
    user_items_weight = defaultdict(list)
   
    for user, items in user_items.items():
        # Calculate TF-IDF score: (1/|user_items|) * log((N+1)/item_freq)
        # where N is the number of neighbors + 1 for smoothing
        interaction_counts = np.array(
            [(1 / len(items)) * np.log((number_nb + 1) / result_dict[item]) for item in items],
            dtype=float
        )
        
        # Normalize weights so they sum to 1 for each user
        sum_values = np.sum(interaction_counts)
        interaction_counts_normalized = interaction_counts / sum_values
        
        # Store normalized weights for each (user, item) pair
        for i, item in enumerate(items):
            weight = interaction_counts_normalized[i]
            user_items_weight[user].append((item, weight))
    
    return dict(user_items_weight)


# =============================================================================
# Knowledge Distillation Loss Function
# =============================================================================
def compute_distillation_loss(student_output, teacher_output, temperature=1.0):
    """
    Compute KL-divergence based distillation loss between student and teacher model outputs.
    
    Args:
        student_output: Logits from the student (local) model
        teacher_output: Logits from the teacher (global) model
        temperature: Temperature parameter for softening probability distributions
        
    Returns:
        distillation_loss: Scaled KL divergence loss
    """
    # Apply log-softmax to student outputs and softmax to teacher outputs
    # Temperature scaling softens the probability distribution for better knowledge transfer
    student_prob = torch.log_softmax(student_output / temperature, dim=1)
    teacher_prob = torch.softmax(teacher_output / temperature, dim=1)
    
    # Compute KL divergence loss and scale by temperature^2 as per distillation literature
    distillation_loss = torch.nn.functional.kl_div(
        student_prob, teacher_prob, reduction='batchmean'
    ) * (temperature ** 2)
    return distillation_loss


# =============================================================================
# Model Persistence Utilities for Client Models
# =============================================================================
def get_client_model_path(client_id, dataset, model_dir):
    """
    Generate a hierarchical file path for storing client model checkpoints.
    Organizes models into folders of 1000 clients each for better file system management.
    
    Args:
        client_id: Unique identifier for the client
        dataset: Name of the dataset being used
        model_dir: Base directory for model storage
        
    Returns:
        Full file path for the client's model checkpoint
    """
    # Create folder index based on client_id (e.g., clients 0-999 go to folder "0-999")
    folder_index = client_id // 1000
    folder_name = f"{model_dir}{dataset}/{folder_index * 1000}-{(folder_index + 1) * 1000 - 1}/"
    
    # Ensure the directory exists
    os.makedirs(folder_name, exist_ok=True)
    
    return f"{folder_name}client_{client_id}.pkl"


def load_client_model(model, client_id, dataset, device, model_dir):
    """
    Load a previously saved client model checkpoint if available.
    
    Args:
        model: The model instance to load weights into
        client_id: Client identifier
        dataset: Dataset name
        device: Torch device (CPU/GPU)
        model_dir: Base directory for model storage
    """
    model_path = get_client_model_path(client_id, dataset, model_dir)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No saved model found for client {client_id}. Using initial model.")


def save_client_model(model, client_id, dataset, model_dir="global_client_models_"):
    """
    Save the current state of a client model to disk.
    
    Args:
        model: The model instance to save
        client_id: Client identifier
        dataset: Dataset name
        model_dir: Base directory for model storage
    """
    model_path = get_client_model_path(client_id, dataset, model_dir)
    torch.save(model.state_dict(), model_path)


# =============================================================================
# Client Selection Strategy Based on Performance Classification
# =============================================================================
def select_users_by_performance(strong_users, weak_users, n_fed_client_each_round, strong_user_ratio):
    """
    Select clients for federated training round based on their performance classification.
    Allows controlled sampling of 'strong' vs 'weak' clients to study heterogeneity effects.
    
    Args:
        strong_users: List of client IDs classified as high-performing
        weak_users: List of client IDs classified as low-performing
        n_fed_client_each_round: Total number of clients to sample per round
        strong_user_ratio: Proportion of strong clients to include in the sample
        
    Returns:
        idxs_users: List of selected client IDs for the current round
    """
    # Calculate number of strong/weak clients to sample
    n_strong_users = int(n_fed_client_each_round * strong_user_ratio)
    n_weak_users = n_fed_client_each_round - n_strong_users
    
    # Shuffle both lists to ensure random sampling within each group
    random.shuffle(strong_users) 
    random.shuffle(weak_users)   
    
    # Sample from each group (with boundary check to avoid index errors)
    selected_strong_users = strong_users[:min(n_strong_users, len(strong_users))]
    selected_weak_users = weak_users[:min(n_weak_users, len(weak_users))]
    
    # Combine and return selected clients
    idxs_users = selected_strong_users + selected_weak_users
    return idxs_users


# =============================================================================
# Sparse Matrix to Sparse Tensor Conversion Utility
# =============================================================================
def _convert_sp_mat_to_sp_tensor(X):
    """
    Convert scipy sparse matrix to PyTorch sparse tensor for GPU-efficient operations.
    
    Args:
        X: scipy.sparse matrix (e.g., adjacency matrix)
        
    Returns:
        torch.sparse.FloatTensor: Sparse tensor representation
    """
    coo = X.tocoo()  # Convert to COO format for easy coordinate extraction
    i = torch.LongTensor([coo.row, coo.col])  # Extract row/col indices
    v = torch.from_numpy(coo.data).float()  # Extract values as float tensor
    return torch.sparse.FloatTensor(i, v, coo.shape)


# =============================================================================
# Main Training Loop - Federated Graph Neural Network for Recommendation
# =============================================================================
if __name__ == '__main__':
    # Set random seed for reproducibility across all libraries
    setSeed.init_seed(42)

    # Parse command-line arguments and configure device
    args = parse_args()
    args.device = torch.device('cuda:0')
    args.dataset = '100k'
    
    # Privacy-preserving configuration flags
    args.pi = False           # Enable/disable privacy injection
    args.add_noise = False    # Enable/disable noise addition for DP
    args.noise = 0.1          # Noise scale parameter
    args.epsilon = 0.01       # Privacy budget (epsilon) for differential privacy
    args.pi_number = 0.1      # Parameter for privacy injection mechanism
    
    # Initialize data generator with dataset path and batch configuration
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, args=args)
    data_generator.print_statistics()
    
    # Generate different types of adjacency matrices for graph convolution
    plain_adj, norm_adj, mean_adj, adj_mat = data_generator.get_adj_mat_new()
    
    # Experiment configuration
    algorithm_name = 'global_test_100k_onlybce'  # Identifier for result logging
    args.mode = 'fednebmask'                      # Federated learning mode with neighbor masking
    args.epoch = 2000                             # Total global communication rounds
    args.local_epoch = 5                          # Local training epochs per client per round
    args.layers = 2                               # Number of GNN layers
    args.embed_size = 64                          # Embedding dimension
    args.lr = 0.001                               # Learning rate for optimizer
    args.alpha = 1                                # Weight for distillation loss (if enabled)

    args.attention = True      # Enable attention mechanism for neighbor aggregation
    args.number_neb = 5        # Number of similar neighbors to consider per user
    args.Distillation = False  # Enable/disable knowledge distillation from global model
    temperature = 1            # Temperature parameter for distillation
    
    # Pre-compute similar users dictionary for neighbor-based aggregation
    t1 = time()
    similar_users_dict = data_generator.optimized_find_all_similar_users(plain_adj, args.number_neb)
    t2 = time()
    print("find_neb_time:", t2 - t1)
    
    # ========================================================================
    # Client Performance Classification (for heterogeneity simulation)
    # ========================================================================
    t3 = time()
    user_performance = {}
    
    # Simulate client heterogeneity: randomly assign performance flags
    probability_1 = 0    # Probability of being a 'strong' client (currently 0%)
    probability_0 = 1 - probability_1
    
    for user_id in range(data_generator.n_users):
        performance_flag = random.choices([0, 1], weights=[probability_0, probability_1], k=1)[0]
        user_performance[user_id] = performance_flag
    
    # Separate clients into strong/weak groups based on performance flag
    strong_users = [uid for uid, flag in user_performance.items() if flag == 1]
    weak_users = [uid for uid, flag in user_performance.items() if flag == 0]
    
    count_0 = sum(1 for flag in user_performance.values() if flag == 0)
    count_1 = sum(1 for flag in user_performance.values() if flag == 1)
   
    print("User Performance:", user_performance)
    print(f"Number of 0s: {count_0}")
    print(f"Number of 1s: {count_1}")
    t4 = time()
    
    # ========================================================================
    # Model Initialization: Global (full) and Local (lightweight) GNNs
    # ========================================================================
    # Global model: Full-capacity GNN for server-side aggregation
    global_model = GNN2_caiyang(
        data_generator.n_users,
        data_generator.n_items,
        norm_adj,
        plain_adj,
        args,
    ).to(args.device)
   
    # Local model configuration for resource-constrained clients
    args_local = client_args()
    args_local.embed_size = 64
    args_local.layers = 2
    args_local.device = args.device
    args_local.attention = args.attention
    args_local.mode = args.mode
    args_local.number_neb = args.number_neb
    args_local.local_epoch = 5

    # Lightweight client model (may have reduced capacity for efficiency)
    local_model = GNN2_caiyang_light(
        data_generator.n_users,
        data_generator.n_items,
        norm_adj,
        plain_adj,
        args_local,
    ).to(args.device)

    # ========================================================================
    # Federated Training Setup
    # ========================================================================
    t0 = time()
    n_fed_client_each_round = 128  # Number of clients sampled per communication round
    n_client = data_generator.n_users
    cur_best_pre_0, stopping_step = 0, 0
    
    # Initialize optimizer for global model parameters
    optimizer = optim.Adam(global_model.parameters(), lr=args.lr)
    
    # Logging lists for training metrics and evaluation results
    loss_loger, pre_loger10, pre_loger20, rec_loger10, rec_loger20, \
        ndcg_loger10, ndcg_loger20, hit_loger10, hit_loger20 = [], [], [], [], [], [], [], [], []
    
    best_hr_10, best_hr_20, best_ndcg_10, best_ndcg_20 = 0, 0, 0, 0
    training_time = 0.0

    # Initialize mask tensor for privacy-preserving neighbor masking mechanism
    mask = torch.zeros(size=norm_adj.shape, device=args.device)

    begin_time = time()
    
    # Convert sparse adjacency matrices to dense tensors for GPU operations
    # Note: For very large graphs, consider keeping sparse format for memory efficiency
    norm_adj = _convert_sp_mat_to_sp_tensor(norm_adj).to(args.device).to_dense()
    plain_adj = _convert_sp_mat_to_sp_tensor(plain_adj).to(args.device).to_dense()

    # ========================================================================
    # Main Federated Learning Loop
    # ========================================================================
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        model_para_list = []  # Store updated parameters from participating clients
        user_ini_state = copy.deepcopy(global_model.state_dict())['embedding_dict.user_emb']
        user_emb_list = {}  # Track user embedding updates for personalized aggregation
        
        # Sample clients for this communication round
        # Option 1: Uniform random sampling (currently active)
        idxs_users = random.sample(range(0, n_client), n_fed_client_each_round)
        # Option 2: Performance-aware sampling (commented out)
        # idxs_users = select_users_by_performance(strong_users, weak_users, n_fed_client_each_round, select_strong_ratio)

        # --------------------------------------------------------------------
        # Local Training Phase: Each selected client trains on local data
        # --------------------------------------------------------------------
        for idx in idxs_users:
            # Save initial global model state for this client's training
            model_ini = copy.deepcopy(global_model.state_dict())
            global_model_copy = copy.deepcopy(global_model)  # For distillation teacher
            model = global_model  # Default to global model architecture
            local_epoch = args.local_epoch
            
            # Use lightweight model for 'weak' clients to simulate system heterogeneity
            if user_performance[idx] == 0:
                model = local_model
                local_epoch = args_local.local_epoch
                # Initialize lightweight model with global model parameters
                model.embedding_dict['user_emb'].data.copy_(global_model.embedding_dict['user_emb'])
                model.embedding_dict['item_emb'].data.copy_(global_model.embedding_dict['item_emb'])

            # ----------------------------------------------------------------
            # Attention-based Neighbor Aggregation with TF-IDF Weighting
            # ----------------------------------------------------------------
            if args.attention and user_performance[idx] == 1:
                # Gather the target user and their similar neighbors
                users = np.concatenate([[idx], similar_users_dict[idx]])
                
                # Collect all positive items interacted by the user group
                pos_items_a = list(itertools.chain.from_iterable(
                    data_generator.train_items[user] for user in users 
                    if user in data_generator.train_items
                ))
                
                # Count item frequencies across the neighbor group
                counter_dict = Counter(pos_items_a)
                result_dict = dict(counter_dict)
                
                # Extract interaction lists for TF-IDF weighting
                user_items = {
                    id: data_generator.train_items[id] 
                    for id in users if id in data_generator.train_items
                }
                
                # Compute TF-IDF weights to prioritize informative items
                user_items_weight = TF_IDF_weight(result_dict, user_items, args.number_neb)
                
                # Apply weights to adjacency matrix edges for attention-aware propagation
                for user, items_weights in user_items_weight.items():
                    for item, weight in items_weights:
                        norm_adj[user, data_generator.n_users + item] = (
                            plain_adj[user, data_generator.n_users + item] * weight
                        )

            # ----------------------------------------------------------------
            # Local Epoch Training Loop
            # ----------------------------------------------------------------
            for i in range(local_epoch):
                # Sample mini-batch: target user, positive items, negative items
                users, pos_items, neg_items = data_generator.sample_local(idx, similar_users_dict)
                
                # Knowledge distillation: generate soft labels from global teacher model
                if args.Distillation:
                    with torch.no_grad():
                        teacher_output = global_model_copy.generate_soft_labels(
                            users, pos_items, neg_items, temperature
                        )
                
                # Forward pass through GNN model
                if args.mode == 'fedneb':
                    # Standard federated neighbor embedding mode
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                        users, pos_items, neg_items, 0, norm_adj
                    )
                else:
                    # Federated neighbor embedding with privacy-preserving mask
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, mask = model(
                        users, pos_items, neg_items, mask, norm_adj
                    )
                
                # Compute BPR-style pairwise ranking loss
                batch_loss, batch_mf_loss, batch_emb_loss = model.create_cross_entropy_loss(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )
                
                # Add distillation loss if enabled
                if args.Distillation:
                    student_output = model.forward_with_logits(users, pos_items, neg_items)
                    distillation_loss = compute_distillation_loss(
                        student_output, teacher_output, temperature
                    )
                
                # Backpropagation step
                optimizer.zero_grad()
                total_loss = batch_loss
                if args.Distillation:
                    total_loss = batch_loss + args.alpha * distillation_loss
                total_loss.backward()
                optimizer.step()

                # Accumulate loss metrics for logging
                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            # ----------------------------------------------------------------
            # Post-Training: Collect updated parameters for aggregation
            # ----------------------------------------------------------------
            model_aft = copy.deepcopy(model.state_dict())
            # Track individual user embedding updates for personalized aggregation
            user_emb_list[idx] = model_aft['embedding_dict.user_emb'][idx]
            model_para_list.append(model_aft)
            
            # Reset global model to initial state for next client
            global_model.load_state_dict(model_ini)

        # ====================================================================
        # Server Aggregation Phase: Federated Averaging
        # ====================================================================
        w_ = FedAvg(model_para_list)
        
        # Personalized aggregation: preserve individual user embeddings
        for j in user_emb_list:
            user_ini_state[j] = user_emb_list[j]
        w_['embedding_dict.user_emb'] = copy.deepcopy(user_ini_state)
        
        # Update global model with aggregated parameters
        global_model.load_state_dict(w_)
        loss_loger.append(loss.item())

        # ====================================================================
        # Evaluation Phase (every 5 epochs)
        # ====================================================================
        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        # Evaluate global model on test set
        hr10, hr20, ndcg10, ndcg20 = test_model(global_model, data_generator)
        t3 = time()

        # Log evaluation metrics
        hit_loger10.append(hr10)
        hit_loger20.append(hr20)
        ndcg_loger10.append(ndcg10)
        ndcg_loger20.append(ndcg20)

        # Track best performance for early stopping and model saving
        if hr10 > best_hr_10 or ndcg10 > best_ndcg_10 or hr20 > best_hr_20 or ndcg20 > best_ndcg_20:
            best_hr_10 = max(best_hr_10, hr10)
            best_ndcg_10 = max(best_ndcg_10, ndcg10)
            best_hr_20 = max(best_hr_20, hr20)
            best_ndcg_20 = max(best_ndcg_20, ndcg20)

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f],  ' \
                    'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                    (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, hr10, hr20, ndcg10, ndcg20)
            print(perf_str)

        # Save best model checkpoint
        if hr10 == best_hr_10 and args.save_flag == 1:
            torch.save(
                global_model.state_dict(), 
                args.weights_path + str(args.dataset) + '/' + str(algorithm_name) + '.pkl'
            )
            print('save the weights in path: ', args.weights_path + str(args.dataset) + '/' + str(algorithm_name) + '.pkl')

    # ========================================================================
    # Final Results Reporting and Logging
    # ========================================================================
    best_hr_10 = max(hit_loger10)
    idx = hit_loger10.index(best_hr_10)

    final_perf = "Best Iter=[%d]@[%.1f]\thit=[%.5f %.5f], ndcg=[%.5f %.5f]" % \
                 (idx, time() - t0, hit_loger10[idx], hit_loger20[idx], 
                  ndcg_loger10[idx], ndcg_loger20[idx])
    print(final_perf)

    # Prepare result dictionary for experiment tracking
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
    
    # Log results to external file/database
    FileProcess.add_row_test(save_dict)
