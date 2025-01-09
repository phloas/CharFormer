
import datetime
import pathlib
import zipfile
import torch
import time
import argparse
import os
import os.path as osp
import glob
# import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from utils import epoch_time, test_hitratio, test_recall, euclidean_distance
from conf import *
from dataset import *
from model import *
from tools import wrloss as los
from graph import draw
from hyperbolics import RAdam
from multiprocessing import Pool, cpu_count


# def dataset_to_one_hot(dic, alphabet_size):
#     for dset in dic.keys():
#         dic[dset] = index_to_one_hot(dic[dset], alphabet_size=alphabet_size)

class GRU(nn.Module):

    def __init__(self, len_sequence, embedding_size, hidden_size, recurrent_layers, alphabet_size, max_length,
                 dropout=0.0):
        super(GRU, self).__init__()

        self.len_sequence = len_sequence
        self.sequence_encoder = nn.GRU(input_size=alphabet_size, hidden_size=hidden_size, num_layers=recurrent_layers,
                                   dropout=dropout)

        self.readout = nn.Linear(in_features=hidden_size, out_features=embedding_size, device=device)
        self.alphabet_size = alphabet_size
        self.max_length = max_length
        self.device = device

    
    def dataset_to_one_hot(self, sequences):
        matrixs = []
        for sequence in sequences:
            matrixs.append(self.index_to_one_hot(sequence))
        return matrixs

    def index_to_one_hot(self, x):
        # add one row of zeros because the -1 represents the absence of element and it is encoded with zeros
        # print(x.size())
        # 创建one-hot编码矩阵，添加一个额外的行全零向量，用于处理缺失值（如 -1）
        one_hot_matrix = torch.cat((torch.eye(self.alphabet_size, device=device), 
                                    torch.zeros((1, self.alphabet_size), device=device)), dim=0)
        
        # 检查x的值是否有效（在范围内）
        if (x >= self.alphabet_size).any() or (x < -1).any():
            raise ValueError("Input indices must be in the range [-1, alphabet_size-1]")

        # 使用one_hot_matrix根据索引x进行索引，自动处理 -1 为零向量
        one_hot_encoded = one_hot_matrix[x + 1]  # 将 -1 转换为索引 0，其它保持不变
        return one_hot_encoded

    def forward(self, sequence):
        # 获取每个序列的实际长度
        lengths = torch.tensor([len(seq) for seq in sequence])

        padded_x = pad_sequence(sequence, batch_first=True, padding_value=0)

        # 如果 padded_x 的长度小于 max_len，则继续 padding 到 max_len
        if padded_x.size(1) < self.max_length:
            padding_size = self.max_length - padded_x.size(1)
            padded_x = torch.nn.functional.pad(padded_x, (0, padding_size))

        # 如果 padded_x 的长度超过 max_len，则裁剪到 max_len
        elif padded_x.size(1) > self.max_length:
            padded_x = padded_x[:, :self.max_length]

        sequence = self.dataset_to_one_hot(padded_x)
        # sequence = torch.tensor(sequence).to(device)
        # 将列表转换为张量
        one_hot_tensor = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        (B, N, _) = one_hot_tensor.shape
        
        packed_sequence = pack_padded_sequence(one_hot_tensor, lengths, batch_first=True, enforce_sorted=False)
    
        # 将 packed_sequence 输入到 GRU 进行编码
        packed_output, _ = self.sequence_encoder(packed_sequence)
        
        # 解包成填充后的输出，以便后续处理
        enc_sequence, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0)
        
        # 取最后一个有效时间步的输出 (使用序列实际长度来获取)
        # 由于序列长度不一，需要用索引来提取每个序列的最后一个有效时间步
        enc_sequence = torch.stack([enc_sequence[i, lengths[i] - 1] for i in range(len(lengths))])
        
        # 通过 readout 层
        enc_sequence = self.readout(enc_sequence)
        # sequence = sequence.transpose(0, 1)
        # enc_sequence, _ = self.sequence_encoder(sequence)
        # enc_sequence = enc_sequence[-1]
        # enc_sequence = self.readout(enc_sequence)
        return enc_sequence
    

def distance(xq, xb):
    def _distance(xq, xb):
        start_time = time.time()
        jobs = Pool().imap(euclidean_distance, zip(xq, [xb for _ in xq]))
        dist = list(tqdm(jobs, total=len(xq), desc="# hamming counting"))
        print("# CGU euclidean distance time: {}".format(time.time() - start_time))
        return np.array(dist).reshape((len(xq), len(xb)))

    if len(xq) < len(xb):
        return _distance(xb, xq).T
    else:
        return _distance(xq, xb)

def l2_dist(q: torch.Tensor, x: torch.Tensor):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == x.shape[1]
    x = x.T
    sqr_q = torch.sum(q ** 2, dim=1, keepdim=True)
    sqr_x = torch.sum(x ** 2, dim=0, keepdim=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[l2 < 0] = 0.0
    return torch.sqrt(l2)

def intersect(gs, ids):
    rc = np.mean([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])
    return rc   

def ranking_recalls(sort, gt):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(sort[0]))))]
    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        print("%6d \t %6d \t" % (t, len(sort[0, :t])), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], sort[:, :t])
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()

# 正负采样
def train(model, best_loss, data_loader, optimizer, criterion):
    print('\nModel training.\n')
    best_epoch = 0
    # #动态调整学习率
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                             verbose=True,
    #                                             factor=factor,
    #                                             patience=patience)
    train_losses, valid_losses = [], []
    early_stop_counter = 0
    print(data_loader.epochs)
    beta = data_loader.beta
    model_dir = f'saved/{data_loader.dataset}/{sampling_num}'
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(data_loader.epochs):
        # 设置成训练模式
        model.train()
        epoch_loss = 0
        agg_r = 0
        agg_m = 0
        batch_idx = 0
        start_time = time.time()
        train_dataloader = data_loader.pos_neg_sample_data_generator(epoch)
        for anchor_input, pos_input, neg_input, pos_distance, neg_distance, pos_neg_distance in train_dataloader:
            
            optimizer.zero_grad()
            anchor_output = model(anchor_input)
           
            pos_output = model(pos_input)

            neg_output = model(neg_input)

            anchor_pos_output = euclidean_distance(anchor_output, pos_output)
            anchor_pos_output = torch.exp(anchor_pos_output * -beta)

            anchor_neg_output = euclidean_distance(anchor_output, neg_output)
            anchor_neg_output = torch.exp(anchor_neg_output * -beta)
            
            pos_neg_output = euclidean_distance(pos_output, neg_output)
            pos_neg_output = torch.exp(pos_neg_output * -beta)

            r, m, loss = criterion(anchor_pos_output, pos_distance, anchor_neg_output, neg_distance, pos_neg_output, pos_neg_distance, epoch)

            loss.backward()
            #对模型的梯度进行裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            epoch_loss += loss.item()
            agg_r += r.item()
            agg_m += m.item()
           
            print('step :', round((batch_idx / (data_loader.ntrain / batch_size)) * 100, 2), '% , batch train loss :', loss.item())
            print('rank loss :', r.item())
            print('mse loss :', m.item())
            batch_idx += 1

        # train_loss是一整个epoch的训练loss
        train_loss = epoch_loss
        # train_loss = agg_r + agg_m
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        train_losses.append(epoch_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            early_stop_counter = 0
            best_epoch = epoch + 1
        else:
            if epoch > data_loader.epochs:
                early_stop_counter += 1
                    # 保存当前模型
        model_path = f'saved/GRU/{data_loader.dataset}/model-{data_loader.epochs}-{epoch + 1}-{train_loss:.3f}.pt'
        old_models = glob.glob(f'saved/GRU/{data_loader.dataset}/model-{data_loader.epochs}-{epoch + 1}-*.pt')
        # 删除旧模型文件
        for old_model in old_models:
            os.remove(old_model)
        torch.save(model.state_dict(), model_path)

        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_loss:.3f}')
        print(f'\tRank Loss: {agg_r:.3f}')
        print(f'\tMse Loss: {agg_m:.3f}')
        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch + 1}!')
            print(f'Best epoch at {best_epoch}!')
            break 
    return best_epoch, round(best_loss, 3)

def test(model, data_loader, criterion, args):
    start_time = time.time()
    query_dataloader = data_loader.test_data_generator()
    base_dataloader = data_loader.base_data_generator()
    
    beta = data_loader.beta
    model.eval()
    print('query embedding begin!')
    embed_query = embed_strings(query_dataloader, model, device)
    print('query embedding end!')

    print('base embedding begin!')
    embed_base = []
    index = 0
    for sequences in base_dataloader:
        # print(index)
        embeded = model(sequences)
        embed_base.append(embeded)
        index += 1
    
    embedded_base = torch.cat(embed_base, axis=0).to(device)
    epoch_loss = 0

    print('distance calculate begin!')
    dist = l2_dist(embed_query, embedded_base)
    print('distance calculate end!')
    dist = torch.tensor(dist)
    pred = torch.exp(-2.2 * dist).to(device)
    true = torch.tensor(data_loader.query_normalized_dist).to(device)
    print('hitratio: ')
    hitratio = test_hitratio(true, pred)
    dist = dist.cpu().numpy()
    sort = np.argsort(dist)
    ranking_recalls(sort, data_loader.query_knn)

def embed_strings(loader, model, device):
    embedded_list = []

    for sequences, dist in loader:
        embeded = model(sequences)
        embedded_list.append(embeded)
    
    embedded_reference = torch.cat(embedded_list, axis=0).to(device)
    return embedded_reference

def run(best_loss, args, data_loader):
    model = GRU(len_sequence = data_loader.maxlength, 
                embedding_size=128, 
                hidden_size=128, 
                recurrent_layers=1, 
                alphabet_size=len(data_loader.vocab),
                max_length=data_loader.maxlength
                ).to(device)

    optimizer = Adam(params=model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
                eps=adam_eps)

    # MSELoss
    triplet_loss = los.TripletLoss(args.epochs)
    criterion = nn.MSELoss().float()

    # best_model, best_loss = train(model, best_loss, data_loader, optimizer, triplet_loss)

    best_model = 83
    best_loss = 14.154
    model.load_state_dict(torch.load('saved/GRU/{}/model-{}-{}-{}.pt'.format(args.dataset, args.epochs, best_model, best_loss)))
    # model.load_state_dict(torch.load('saved/model-{}-{}-{}.pt'.format(args.dataset, sampling_num, args.epochs)))
    print('\nModel evaluation.')

    test_loss = test(model, data_loader, criterion, args)

def get_args():
    parser = argparse.ArgumentParser(description='HyperParameters for CharTransformer')
    parser.add_argument("--dataset", type=str, default="uniref_name", help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nv", type=int, default=1000, help="# of valid items")
    parser.add_argument('--mode', type=str, default='train', help='Type of train and test')
    parser.add_argument('--test-all', action="store_true", default=False, help='test multiple models at once')
    parser.add_argument("--embed", type=str, default="trans", help="embedding method")
    parser.add_argument("--epochs", type=int, default=100, help="# number of epochs")
    parser.add_argument("--nheads", type=int, default=4, help="# number of multiheads")
    parser.add_argument('--distance', type=str, default='square', help='Type of distance to use')
    parser.add_argument('--scaling', type=str, default='False', help='Project to hypersphere (for hyperbolic)')
    parser.add_argument('--hyp_optimizer', type=str, default='Adam', help='Optimizer for hyperbolic (Adam or RAdam)')
    parser.add_argument("--save-split", action="store_true", default=False, help="save split data folder")
    parser.add_argument("--learn", action="store_true", default=False, help="learnable parameter")
    parser.add_argument("--recall", action="store_true", default=False, help="print recall")
    parser.add_argument("--augment", action="store_true", default=False, help="data augmentation")
    parser.add_argument("--sub", action="store_true", default=False, help="sub-string")
    parser.add_argument("--twosample", action="store_true", default=False, help="twosample")
    parser.add_argument("--mask", type=str, default="global",help="# mask")
    # parser.add_argument("--sample", type=int, default=5, help="# of sampling number")
    args = parser.parse_args()

    if args.dataset == 'word': 
        file_path = 'preprocess/{}/word.txt'.format(args.dataset)
        beta = 1.80
    elif args.dataset == 'uniref_300':
        file_path = 'preprocess/uniref/{}.txt'.format(args.dataset)
        beta = 0.78
    elif args.dataset == 'uniref_500':
        file_path = 'preprocess/uniref/{}.txt'.format(args.dataset)
        beta = 0.80
    elif args.dataset == 'gen50ks_500':
        file_path = 'preprocess/gen50ks/{}.txt'.format(args.dataset)
        beta = 0.93
    elif args.dataset == 'uniref_name':
        file_path = 'preprocess/uniref/protein_names.txt'
        beta = 2.17
    elif args.dataset == 'DBLP':
        file_path = 'preprocess/DBLP/title.txt'
        beta = 2.14
    elif args.dataset == 'uniprot':
        file_path = '.data/open_source_datasets/uniprot/uniprot_300.txt'
        beta = 0.89
    data_loader = Dataset(args= args, file_path = file_path, beta = beta)
    data_loader
    if args.save_split:
        data_loader.save_split()
    return args, data_loader

if __name__ == '__main__':
    args, data_loader = get_args()
    best_val_loss = float('inf')
    run(best_loss = inf, args = args, data_loader = data_loader)