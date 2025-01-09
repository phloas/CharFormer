'''
Author: B34
Date: 2023-11-21 15:49:13
Description: 
'''

import datetime
import pathlib
import zipfile
import torch
import time
import argparse
import os
import os.path as osp
import glob

import torch.nn.functional as F
from torch.optim import Adam
from utils import epoch_time, zipdir, test_hitratio, test_recall
from conf import * 
from dataset import *
from model import *
from tools import wrloss as los
from hyperbolics import RAdam

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN or infinite values found in gradients of {name}")
                raise ValueError("NaN or infinite values in gradients")

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def write_txt(i, output, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('step :' + str(i) + str(output) + '\n')

def char_to_index(sentence):
    return [ord(char) for char in sentence]

def compute_euclidean(src_output, tar_output, beta):
    return torch.exp(-beta * euclidean_distance(src_output, tar_output))

def compute_squared_euclidean(src_output, tar_output, beta):
    return torch.exp(-beta * euclidean_distance_squared(src_output, tar_output))

def compute_cosine(src_output, tar_output):
    return F.cosine_similarity(src_output, tar_output, dim = 1)

def concat(src_output, tar_output):
    combined_output = torch.cat((src_output, tar_output),dim = 1)
    return combined_output

# 正负采样
def train(model, best_loss, data_loader, optimizer, criterion):
    print('\nModel training.\n')
    best_epoch = 0
    milestones_list = [100, 125, 150, 175]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)
    train_losses, valid_losses = [], []
    early_stop_counter = 0
    print(data_loader.epochs)
    beta = data_loader.beta
    model_dir = f'saved/{data_loader.dataset}/{sampling_num}'
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(epochs):
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
            anchor_pos_output = model(anchor_input, pos_input,beta)

            anchor_neg_output = model(anchor_input, neg_input,beta)

            pos_neg_output = model(pos_input, neg_input,beta)

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

        scheduler.step()
        print(f'Used learning rate:{scheduler.get_last_lr()[0]}')
        # 验证集部分
        # start_time = time.time()
        # valid_loss = evaluate(model, data_loader) 
        # end_time = time.time()
        # valid_mins, valid_secs = epoch_time(start_time, end_time)
        # valid_losses.append(valid_loss)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     early_stop_counter = 0
        #     best_epoch = epoch + 1
        #     torch.save(model.state_dict(), 'saved/model-{}-{}-{}.pt'.format(args.dataset, sampling_num, data_loader.epochs))
        # else:
        #     if epoch > data_loader.epochs:
        #         early_stop_counter += 1
        
        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     early_stop_counter = 0
        #     best_epoch = epoch + 1
        #     torch.save(model.state_dict(), 'saved/model-{}-{}-{}.pt'.format(args.dataset, sampling_num, data_loader.epochs))
        # else:
        #     if epoch > data_loader.epochs:
        #         early_stop_counter += 1

        if train_loss < best_loss:
            best_loss = train_loss
            early_stop_counter = 0
            best_epoch = epoch + 1
        else:
            if epoch > data_loader.epochs:
                early_stop_counter += 1
        model_path = f'saved/{data_loader.dataset}/{sampling_num}/model-{data_loader.epochs}-{epoch + 1}-{train_loss:.3f}.pt'
        old_models = glob.glob(f'saved/{data_loader.dataset}/{sampling_num}/model-{data_loader.epochs}-{epoch + 1}-*.pt')
        for old_model in old_models:
            os.remove(old_model)
        torch.save(model.state_dict(), model_path)
        f = open('result/{}/train_loss.txt'.format(args.dataset), 'w')
        f.write(str(train_losses))
        f.close()
        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_loss:.3f}')
        print(f'\tRank Loss: {agg_r:.3f}')
        print(f'\tMse Loss: {agg_m:.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f}')
        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch + 1}!')
            print(f'Best epoch at {best_epoch}!')
            break 
    return best_epoch, round(best_loss, 3)

def evaluate(model, data_loader):
    criterion = nn.MSELoss().float()
    model.eval()
    valid_dataloader = data_loader.valid_data_generator()
    beta = data_loader.beta
    valid_loss = 0
    with torch.no_grad():
        batch_idx = 1
        for valid_input, distance in valid_dataloader:
            for index, (pad_input) in enumerate(data_loader.valid_sequence):
                pad_input = torch.unsqueeze(pad_input, dim = 0)
                pad_input = pad_input.repeat(len(valid_input), 1)
                valid_output = model(valid_input, pad_input, beta)
                current_column = distance[:,index]
                loss = criterion(valid_output, current_column)
                valid_loss += loss.item()
            
            batch_idx += 1
    return valid_loss / (data_loader.nvalid / batch_size_valid * 100)


def test(model, data_loader, criterion, args):
    start_time = time.time()
    test_dataloader = data_loader.test_data_generator()
    beta = data_loader.beta
    model.eval()
    epoch_loss = 0
    batch_idx = 0
    with torch.no_grad():
        total_result = torch.zeros(0,data_loader.nbase).to(device)
        for input, distance in test_dataloader:
            length = len(data_loader.query_ids)
            batch_sum = length // batch_size_valid + (length % batch_size_valid > 0)
            res_list = []
            for index, (base_input) in enumerate(data_loader.pad_base):
                print('batch:', batch_idx + 1)
                print('sum_batch:', batch_sum)
                base_input = torch.unsqueeze(base_input, dim = 0)
                base_input = base_input.repeat(len(input), 1)
                output = model(input, base_input, beta)
                current_column = distance[:,index]
                res_list.append(output)
                result = torch.stack(res_list, dim = 1)
                loss = criterion(output, current_column)
                epoch_loss += loss.item()
                print('step : ', round((index / data_loader.nbase) * 100, 2), '% , batch loss :', loss.item())
            total_result = torch.cat((total_result, result), dim = 0)
            batch_idx += 1
        
        query_normalized_dist = torch.tensor(data_loader.query_normalized_dist).to(device)
        hit_ratio = test_hitratio(query_normalized_dist, total_result)
        if args.recall:
            result = torch.tensor(total_result).cpu()
            result = -result.numpy()
            test_recall(result, data_loader.query_knn)
        end_time = time.time()
        test_loss = epoch_loss / (batch_sum * batch_size_valid)
        print(f'\tTest Loss: {test_loss:.3f}')
        test_mins, test_secs = epoch_time(start_time, end_time)
        print(f'Test Time: {test_mins}m {test_secs}s')
    return test_loss

def run(best_loss, args, data_loader):
    model = Cross_Bi_Encoder(src_pad_idx = 0,
                vocab_size = len(data_loader.vocab),
                max_len = data_loader.maxlength,
                distance = args.distance,
                scaling = args.scaling,
                mask = args.mask
                ).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)

    if args.distance == 'hyperbolic' and args.hyp_optimizer == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=init_lr)
    else:
        optimizer = Adam(params=model.parameters(),
                    lr=init_lr,
                    weight_decay=weight_decay,
                    eps=adam_eps)

    triplet_loss = los.TripletLoss(args.epochs)
    criterion = nn.MSELoss().float()
    res_path = 'result/res.txt'
    best_model, best_loss = train(model, best_loss, data_loader, optimizer, triplet_loss)
    model.load_state_dict(torch.load('saved/{}/{}/model-{}-{}-{}.pt'.format(args.dataset, sampling_num, args.epochs, best_model, best_loss)))
    print('\nModel evaluation.')
    test_loss = test(model, data_loader, criterion, args)

def test_all(args, data_loader):
    
    model = Cross_Bi_Encoder(src_pad_idx = 0,
            vocab_size = len(data_loader.vocab),
            max_len = data_loader.maxlength,
            distance = args.distance,
            scaling = args.scaling,
            mask = args.mask
            ).to(device)
    criterion = nn.MSELoss().float()


    model_dir = f'saved/{args.dataset}/{sampling_num}'
    model_files = [f'saved/{args.dataset}/{sampling_num}/model-{args.epochs}-143-16.673.pt']
   
    for model_file in model_files:
        model.load_state_dict(torch.load(model_file))
        print('model_file', model_file)
        test(model, data_loader, criterion, args)

def get_args():
    parser = argparse.ArgumentParser(description='HyperParameters for CharTransformer')
    parser.add_argument("--dataset", type=str, default="word", help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nv", type=int, default=1000, help="# of valid items")
    parser.add_argument('--mode', type=str, default='train', help='Type of train and test')
    parser.add_argument('--test-all', action="store_true", default=False, help='test multiple models at once')
    parser.add_argument("--embed", type=str, default="trans", help="embedding method")
    parser.add_argument("--epochs", type=int, default=90, help="# number of epochs")
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
    elif args.dataset == 'uniref_name':
        file_path = 'preprocess/uniref/protein_names.txt'
        beta = 2.17
    data_loader = Dataset(args= args, file_path = file_path, beta = beta)
    if args.save_split:
        data_loader.save_split()
    return args, data_loader

if __name__ == '__main__':

    args, data_loader = get_args()
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = osp.join('log', args.dataset, current_time)
    if not osp.isdir(log_path):
        os.makedirs(log_path)
    # logger = Logger(log_path).print_log()
    zipf = zipfile.ZipFile(os.path.join(log_path, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    best_val_loss = float('inf')
    if args.test_all:
        test_all(args, data_loader)
    else:
        run(best_loss = inf, args = args, data_loader = data_loader)

