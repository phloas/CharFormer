import torch
import time
import Levenshtein
import os
import numpy as np
import math
from conf import *
from tqdm import tqdm
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence

# compute spent time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# compute max length
def max_length_in_txt(file_path):
    max_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            current_length = len(line.strip())
            max_length = max(max_length, current_length)
    return max_length

def f(x):
    a, B = x
    return [Levenshtein.distance(a, b) for b in B]

def all_pair_distance(A, B, n_thread, progress=True):
    bar = tqdm if progress else lambda iterable, total, desc: iterable

    def all_pair(A, B, n_thread):
        with Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(
                bar(
                    pool.imap(f, zip(A, [B for _ in A])),
                    total=len(A),
                    desc="# edit distance {}x{}".format(len(A), len(B)),
                ))
            if progress:
                print("# Calculate edit distance time: {}".format(time.time() - start_time))
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)

def extract_non_padding_values(seq):
    idx = (seq == 0).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return seq
    return seq[:idx[0]]

def padsequence(sequences, max_length, device):
    padded_sequences = [seq[:max_length] if len(seq) > max_length else torch.cat([seq, torch.zeros(max_length - len(seq), dtype = torch.int)]) for seq in sequences]
    return pad_sequence(padded_sequences, batch_first=True, padding_value=0).to(device)

def pad(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[:max_len]
    else:
        return torch.cat([sequence,torch.zeros(max_len - len(sequence), dtype = torch.int).to(device)])
    
def euclidean_distance(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    distance = torch.norm(tensor1 - tensor2, p=2, dim=1)
    return distance

def euclidean_distance_squared(tensor1, tensor2):
    
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    return torch.sum((tensor1 - tensor2) ** 2, dim=-1)

def safe_acosh(x):
    """ 
    Computes the arccosh of x in a numerically stable way.
    Ensures the input is greater than or equal to 1.
    """
    return torch.log(x + torch.sqrt(x**2 - 1 + 1e-7))

def poincare_distance(x, y, eps=1e-5):

    assert x.shape == y.shape, "x and y must have the same shape"

    norm_x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)

    norm_x_sq = torch.clamp(norm_x_sq, max=1 - eps)
    norm_y_sq = torch.clamp(norm_y_sq, max=1 - eps)

    sq_euclidean_dist = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    num = 2 * sq_euclidean_dist
    denom = (1 - norm_x_sq) * (1 - norm_y_sq) + eps 
    
    argument = 1 + num / denom
    argument = torch.clamp(argument, min=1 + eps)

    dist = safe_acosh(argument)

    return dist.squeeze()

def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    # sort_id_mat = torch.argsort(data, descending=False)
    sort_id_mat = torch.argsort(data, descending=True)
    n = sort_id_mat.shape[0]
    # Tie inclusive.
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k], k
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k], k

def prec_at_ks(true_r, pred_r, ks, rm = 0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    hit_num = 0
    total_num = 0
    for i in range(len(true_r)):
        row_true = true_r[i]
        row_pred = pred_r[i]
        true_ids, true_k = top_k_ids(row_true, ks, inclusive=True, rm=rm)
        print(true_ids)
        max_output, max_indices = torch.topk(row_pred, k = true_k , largest = False)
        print(max_indices)
        hit_num += torch.sum(torch.isin(true_ids, max_indices))
        total_num += true_k
    return hit_num, total_num


def intersect_sizes(gs, ids):
    return np.array([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])

def arg_sort(data):
    sort_id_mat = torch.argsort(-data, dim = 1)
    return sort_id_mat

def test_recall(pred_r, knn):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(knn[0]))))]
    sort_idx = np.argsort(pred_r)
    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        ids = sort_idx[:,:t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="") 
        tps = [intersect_sizes(knn[:, :top_k], ids) / float(top_k) for top_k in ks]
        rcs = [np.mean(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        print()
   

def estimation_error(true_r, pre_r):
    result = torch.abs(pre_r - true_r) / true_r
    return torch.mean(result)

def test_hitratio(true_r, pred_r, rm = 0):
    ks = [1, 5, 10, 50, 100]
    hit_num = torch.zeros(len(ks)).to(device)
    hit_ratio = torch.zeros(len(ks)).to(device)

    save_file = 'top_indices.txt'
    if not os.path.exists(save_file):
        open(save_file, 'w').close()

    with open(save_file,'w') as f:
        for i in range(len(pred_r)):
            row_true = true_r[i]
            row_pred = pred_r[i]
            j = 0
            for top_k in ks:
                true_ids, true_k = top_k_ids(row_true, top_k, inclusive=True, rm=rm)
                max_true_output, max_true_ids = torch.topk(row_true, k = true_k, largest= True)
                max_output, max_indices = torch.topk(row_pred, k = true_k , largest = True)
                if top_k in [1, 5, 10]:
                    f.write(f"Sample {i}, Top-{top_k} True IDs: {true_ids.tolist()}\n")
                    f.write(f"Sample {i}, Top-{top_k} True outputs: {max_true_output.tolist()}\n")
                    f.write(f"Sample {i}, Top-{top_k} Predicted Indices: {max_indices.tolist()}\n")
                    f.write(f"Sample {i}, Top-{top_k} Predicted outputs: {max_output.tolist()}\n\n")
                if torch.sum(torch.isin(true_ids, max_indices)) <= top_k:
                    hit_num[j] += torch.sum(torch.isin(true_ids, max_indices))
                else:
                    hit_num[j] += top_k
                j += 1
    for i in range(len(ks)):
        hit_ratio[i] = hit_num[i] / (len(pred_r) * ks[i])
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for ratio in hit_ratio:
        print("%.4f \t" % ratio, end="")
    print()
    return hit_ratio

def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)