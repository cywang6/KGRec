from .metrics import *
from .parser import parse_args_kgsr

import torch
import numpy as np
import multiprocessing
import heapq
from time import time

import pickle

cores = multiprocessing.cpu_count() // 2

args = parse_args_kgsr()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

############################
# Define our specific items
############################
special_items = [
    'agis_os05g045150', 'agis_os11g015540', 'agis_os11g038210',
    'agis_os08g034710', 'agis_os03g056190', 'agis_os03g024270',
    'agis_os08g038440', 'agis_os06g014910', 'agis_os12g016170',
    'agis_os12g039280', 'agis_os01g002460', 'agis_os07g002940',
    'agis_os04g042480', 'agis_os09g033350', 'agis_os03g015160',
    'agis_os01g048330', 'agis_os04g032850', 'agis_os05g034560',
    'agis_os08g039100'
]
# load entity2id
entity2id, id2entity, id2pheno = {}, {}, {}
with open(args.data_path + args.dataset + '/entity2id.txt', 'r') as f:
    for line in f:
        entity, idx = line.strip().split('\t')
        entity2id[entity] = int(idx)
        id2entity[int(idx)] = entity

with open(args.data_path + args.dataset + '/pheno2id.txt', 'r') as f:
    for line in f:
        pheno, idx = line.strip().split(' ')
        id2pheno[int(idx)] = pheno

special_item_indices = [entity2id[item] for item in special_items]


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

# Self added to get top K items (genes) for each user (phenotype)
def get_top_K_items(test_items, rating, K):
    item_score = {}
    for i in test_items:
        item_score[id2entity[i]] = rating[i]
    K_max_item_score = heapq.nlargest(K, item_score.items(), key=lambda x: x[1])
    return K_max_item_score

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    result = get_performance(user_pos_test, r, auc, Ks)

    # AUC on training set
    testing_items = test_user_set[u]
    user_pos_train = train_user_set[u]
    all_items = set(range(0, n_items))
    train_items = list(all_items - set(testing_items))
    _, auc_train = ranklist_by_sorted(user_pos_train, train_items, rating, Ks)
    result['auc_train'] = auc_train

    ########################################
    # Collect scores for the special items
    ########################################
    user_item_scores = {}
    for idx in special_item_indices:
        user_item_scores[id2entity[idx]] = float(rating[idx].item() if hasattr(rating[idx], 'item') 
                                        else rating[idx])
    result['special_item_scores'] = user_item_scores
    # Save top K items for each user (phenotype)
    K_max = 1000
    result['top_100_items'] = get_top_K_items(all_items, rating, K_max)
    ########################################

    return result


def test(model, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.,
              'auc_list': [],
              'auc_train': 0.,
              'special_item_scores': {},
              'top_K_items': {}
            }

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    print('n_users: ', n_users)
    print('n_items: ', n_items)

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    entity_gcn_emb, user_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)


        for re, user_id in zip(batch_result, user_list_batch):
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            result['auc_list'].append((user_id, re['auc']))
            result['auc_train'] += re['auc_train']/n_test_users

            # Store the special item scores in the result
            result['special_item_scores'][id2pheno[user_id]] = re['special_item_scores']
            if id2pheno[user_id] == 'photoperiod_sensing' or id2pheno[user_id] == 'photosynthetic_efficiency':
                result['top_K_items'][id2pheno[user_id]] = re['top_K_items']

    assert count == n_test_users
    pool.close()
    return result
