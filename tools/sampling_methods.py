import random
import torch
import numpy as np
from conf import *

def random_sampling(train_seq_len, index):
    sampling_index_list = random.sample(range(train_seq_len), sampling_num)
    return sampling_index_list

def pos_neg_CNNED(knn, ntrain, index):
    pre_sort = list(knn[index])[:K]
    pos_sample_index = []
    neg_sample_index = []
    for _ in range(sampling_num):
        sample_pair = random.sample(pre_sort[1:],2)
        if pre_sort.index(sample_pair[0]) < pre_sort.index(sample_pair[1]):
            pos_sample_index.append(sample_pair[0])
            neg_sample_index.append(sample_pair[1])
        else:
            pos_sample_index.append(sample_pair[1])
            neg_sample_index.append(sample_pair[0])
    return pos_sample_index, neg_sample_index

def pos_neg_distance_sampling_1000_half(knn, ntrain, index):
    pre_sort = list(knn[index])
    half_dist = int(ntrain / 2)
    if half_dist % 2 != 0:
        half_dist += 1
    pos_sample_index = random.sample(pre_sort[1:half_dist], sampling_num)
    neg_sample_index = random.sample(pre_sort[half_dist:], sampling_num)
    return pos_sample_index, neg_sample_index

def pos_neg_distance_sampling_knn_half(knn, ntrain, index):
    pre_sort = list(knn[index])[:K]
    half_dist = int(K / 2)
    if half_dist % 2 != 0:
        half_dist += 1
    pos_sample_index = random.sample(pre_sort[1:half_dist], sampling_num)
    neg_sample_index = random.sample(pre_sort[half_dist:], sampling_num)
    return pos_sample_index, neg_sample_index

def pos_neg_bio_kNN(knn, ntrain, index):
    ## 划分成3块区域 
    # 确保每一个pos都会比neg要小。
    pos_sample_index = []
    neg_sample_index = []
    pre_sort = list(knn[index])
    sample_index = random.sample(pre_sort[K:], 2)
    if pre_sort.index(sample_index[0]) < pre_sort.index(sample_index[1]):
        pos_sample_index.append(sample_index[0])
        neg_sample_index.append(sample_index[1])
    else:
        pos_sample_index.append(sample_index[1])
        neg_sample_index.append(sample_index[0])
    pos_sample_index.extend(random.sample(pre_sort[1:K], 1))
    neg_sample_index.extend(random.sample(pre_sort[K:], 1))
    for i in range(sampling_num - 2):
        sample_index = random.sample(pre_sort[1:K], 2)
        if pre_sort.index(sample_index[0]) < pre_sort.index(sample_index[1]):
            pos_sample_index.append(sample_index[0])
            neg_sample_index.append(sample_index[1])
        else:
            pos_sample_index.append(sample_index[1])
            neg_sample_index.append(sample_index[0])
    return pos_sample_index, neg_sample_index

def pos_neg_GRU(knn, ntrain, index):
    pos_sample_index = []
    neg_sample_index = []
    pre_sort = list(knn[index])
    for i in range(sampling_num):
        sample_index = random.sample(pre_sort[1:K], 1)
        pos_sample_index.append(sample_index[0])
        sample_index = random.sample(pre_sort[K:],1)
        neg_sample_index.append(sample_index[0])
    return pos_sample_index, neg_sample_index

def pos_neg_augment_distance_sampling(knn, ntrain, index, prob_augmented = 0.5):
    pre_sort = list(knn[index])
    half_dist = int(ntrain / 2)
    # lens = 100
    # half_dist = int(lens / 2)
    if half_dist % 2 != 0:
        half_dist += 1

    if random.random() < prob_augmented:
        pos_sample_index = random.sample(pre_sort[1:sampling_num + 1], sampling_num)
    else:
        pos_sample_index = random.sample(pre_sort[1:half_dist], sampling_num)
    neg_sample_index = random.sample(pre_sort[half_dist:], sampling_num)
    return pos_sample_index, neg_sample_index

def distance_sampling(distance_data, ntrain, index):
    pre_sort = distance_data[index]
    pre_sort = torch.tensor(pre_sort).to(device)
    # pre_sort = torch.tensor(distance_index[:train_seq_len]).to(device)
    sample_index = []
    t = 0.0
    importance = []
    for i in pre_sort / torch.sum(pre_sort):
        importance.append(t)
        t += i.item()
    importance = torch.tensor(importance).to(device)
    while len(sample_index) < sampling_num:
        a = random.uniform(0,1)
        idx = torch.nonzero(importance > a, as_tuple=True)[0]
        if len(idx) == 0: 
            sample_index.append(ntrain - 1)
        elif ((idx[0]-1) not in sample_index) & ((idx[0]-1) != index):
            sample_index.append(idx[0]-1)

    sorted_sample_index = sorted([(i, pre_sort[i]) for i in sample_index], key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]

def negative_distance_sampling(distance_data, ntrain, index):
    pre_sort = distance_data[index]
    pre_sort = torch.tensor(pre_sort).to(device)
    # pre_sort = torch.tensor(distance_index[:train_seq_len]).to(device)
    pre_sort = torch.ones_like(pre_sort) - pre_sort
    # print [(i,j) for i,j in enumerate(pre_sort)]
    sample_index = []
    t = 0.0
    importance = []
    for i in pre_sort / torch.sum(pre_sort):
        importance.append(t)
        t += i.item()
    importance = torch.tensor(importance).to(device)
    # print importance
    while len(sample_index) < sampling_num:
        a = random.uniform(0,1)
        idx = torch.nonzero(importance > a, as_tuple=True)[0]
        if len(idx) == 0: 
            sample_index.append(ntrain - 1)
        elif ((idx[0]-1) not in sample_index) & ((idx[0]-1) != index): 
            sample_index.append(idx[0]-1)

    sorted_sample_index = sorted([(i, pre_sort[i]) for i in sample_index], key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]
