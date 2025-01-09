'''
Author: B34
Date: 2023-11-21 15:49:06
Description: 
'''

import random
import torch
import tqdm
import time
import string
import Levenshtein
import os
import nlpaug.augmenter.char as nac
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from multiprocessing import cpu_count

from tools import sampling_methods as sm

from conf import *
from utils import max_length_in_txt, all_pair_distance, padsequence
from torch.utils.data import DataLoader, TensorDataset

# def padsequence(sequences, max_length, device):
#     sequences = [torch.tensor(seq) for seq in sequences]
#     padded_sequences = [seq[:max_length] if len(seq) > max_length else torch.cat([seq, torch.zeros(max_length - len(seq), dtype = torch.int)]) for seq in sequences]
#     return pad_sequence(padded_sequences, batch_first=True, padding_value=0).to(device)
    # return pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value= 0).to(device)
    
def char_to_index(sentence, index):
    return [index[char] for char in sentence]

def char_to_pad(sentences, vocab, max_length, device):
    sequence =  [char_to_index(sentence, vocab) for sentence in sentences]
    pad_sequence = padsequence(sequence,max_length, device)
    return pad_sequence

def get_dist_knn(beta, queries, base = None):
    if base is None:
        base = queries
    
    dist = all_pair_distance(queries, base, cpu_count())
    max_dist = np.max(dist)
    normalized_distance = dist / max_dist
    # CNNED的归一化的方式
    avg_dist = np.mean(dist)
    # normalized_distance = dist / avg_dist
    normalized_dist = np.exp(-beta * normalized_distance).astype(np.float32)

    return dist, normalized_dist, get_knn(dist)

def get_knn(dist):
    # 按照升序返回索引
    knn = np.empty(dtype=np.int32, shape=(len(dist), len(dist[0])))
    for i in tqdm.tqdm(range(len(dist)), desc="# sorting for KNN indices"):
        knn[i, :] = np.argsort(dist[i, :])
    return knn

class Dataset:

    def __init__(self, args, file_path, beta):
        self.file_path = file_path
        self.beta = beta
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.maxlength = max_length_in_txt(self.file_path)
        self.maxlength += 2
        self.ntrain = args.nt
        self.nquery = args.nq
        self.nvalid = args.nv
        self.augment = args.augment
        self.sub = args.sub
        self.twosample = args.twosample

        start_time = time.time()
        self.read_dataset()
        print("# Loading time: {}".format(time.time() - start_time))
        self.nlines = len(self.data)
        self.nbase = self.nlines - self.ntrain - self.nquery - self.nvalid

        self.build_vocab()
        self.tokenizer = CharTokenizer(self.vocab, self.maxlength)
        self.load_ids()
        self.load_dist()
        self.pad_data()
        if self.sub:
            self.load_sub()



    def read_dataset(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.data = [line.strip() for line in file.readlines()]
        
    def generate_ids(self):
        np.random.seed(seed)
        idx = np.arange(self.nlines)
        np.random.shuffle(idx)
        print("# shuffled index: ", idx)
        self.train_ids = idx[:self.ntrain]
        self.query_ids = idx[self.ntrain : self.nquery + self.ntrain]
        self.valid_ids = idx[self.nquery + self.ntrain: self.nquery + self.ntrain + self.nvalid]
        self.base_ids = idx[self.nquery + self.ntrain + self.nvalid :]

    # def generate_ids_non(self):
    #     # 不再打乱
    #     idx = np.arange(self.nlines)
    #     print("index: ", idx)

    #     self.train_ids = idx[:self.ntrain]
    #     self.query_ids = idx[self.ntrain : self.nquery + self.ntrain]
    #     self.valid_ids = idx[self.nquery + self.ntrain: self.nquery + self.ntrain + self.nvalid]
    #     self.base_ids = idx[self.nquery + self.ntrain + self.nvalid :]

   
    def generate_dist(self):
        self.train_dist, self.train_normalized_dist, self.train_knn = get_dist_knn(
            self.beta, [self.data[i] for i in self.train_ids]
        )
        self.valid_dist, self.valid_normalized_dist, self.valid_knn = get_dist_knn(
            self.beta, [self.data[i] for i in self.valid_ids]
        )
        self.query_dist, self.query_normalized_dist, self.query_knn = get_dist_knn(
            self.beta,
            [self.data[i] for i in self.query_ids],
            [self.data[i] for i in self.base_ids],
        )

    def load_ids(self):
        idx_dir = "folder/{}/{}/idx/".format(self.dataset, self.ntrain)
        os.makedirs(idx_dir, exist_ok= True)
        if not os.path.isfile(idx_dir + "train_idx.txt"):
            self.generate_ids()
            # self.generate_ids_non()
            np.savetxt(idx_dir + "train_idx.txt", self.train_ids,fmt='%d')
            np.savetxt(idx_dir + "query_idx.txt", self.query_ids,fmt='%d')
            np.savetxt(idx_dir + "base_idx.txt", self.base_ids,fmt='%d')
            np.savetxt(idx_dir + "valid_idx.txt", self.valid_ids,fmt='%d')
        else:
            print("# loading indices from file")
            self.train_ids = np.loadtxt(idx_dir + "train_idx.txt",dtype= int)
            self.query_ids = np.loadtxt(idx_dir + "query_idx.txt",dtype= int)
            self.base_ids = np.loadtxt(idx_dir + "base_idx.txt",dtype= int)
            self.valid_ids = np.loadtxt(idx_dir + "valid_idx.txt",dtype= int)

        print(
            "# Dataset name         : {}".format(self.dataset),
            "# Unique signature     : {}".format(len(self.vocab)),
            "# Maximum length       : {}".format(self.maxlength),
            "# Sampled Train Items  : {}".format(self.ntrain),
            "# Sampled Query Items  : {}".format(self.nquery),
            "# Sampled Valid Items  : {}".format(self.nvalid),
            "# Number of Base Items : {}".format(self.nbase),
            "# Number of Items      : {}".format(self.nlines),
            "# Number of Epochs     : {}".format(self.epochs),
            "# Using sub            : {}".format(self.sub),
            "# Beta                 : {}".format(self.beta),
            "# Number of Layers     : {}".format(n_layers),
            "# Number of Sampling   : {}".format(sampling_num),
            "# Learning rate        : {}".format(init_lr),
            "# Dropout prob         : {}".format(drop_prob),
            "# Sample KNN           : {}".format(K),
            "# Nheads               : {}".format(n_head),
            sep="\n",
        )

    def load_dist(self):
        dis_dir = "folder/{}/{}/dis/".format(self.dataset, self.ntrain)
        os.makedirs(dis_dir, exist_ok= True)
        if not os.path.isfile(dis_dir + "train_dist.txt"):
            self.generate_dist()
            np.savetxt(dis_dir + "train_dist.txt", self.train_dist,fmt='%d')
            np.savetxt(dis_dir + "train_knn.txt", self.train_knn,fmt='%d')
            np.savetxt(dis_dir + "train_normalized_dist.txt", self.train_normalized_dist)
            np.savetxt(dis_dir + "query_dist.txt", self.query_dist,fmt='%d')
            np.savetxt(dis_dir + "query_knn.txt", self.query_knn,fmt='%d')
            np.savetxt(dis_dir + "query_normalized_dist.txt", self.query_normalized_dist)
            np.savetxt(dis_dir + "valid_dist.txt", self.valid_dist,fmt='%d')
            np.savetxt(dis_dir + "valid_knn.txt", self.valid_knn,fmt='%d')
            np.savetxt(dis_dir + "valid_normalized_dist.txt", self.valid_normalized_dist)

        else:
            print("# loading dist and knn from file")
            start_time = time.time()
            self.train_dist = np.loadtxt(dis_dir + "train_dist.txt",dtype= int)
            self.train_knn = np.loadtxt(dis_dir + "train_knn.txt",dtype= int)
            self.train_normalized_dist = np.loadtxt(dis_dir + "train_normalized_dist.txt")
            self.query_dist = np.loadtxt(dis_dir + "query_dist.txt",dtype= int)
            self.query_knn = np.loadtxt(dis_dir + "query_knn.txt",dtype= int)
            self.query_normalized_dist = np.loadtxt(dis_dir + "query_normalized_dist.txt")
            self.valid_dist = np.loadtxt(dis_dir + "valid_dist.txt",dtype= int)
            self.valid_knn = np.loadtxt(dis_dir + "valid_knn.txt",dtype= int)
            self.valid_normalized_dist = np.loadtxt(dis_dir + "valid_normalized_dist.txt")
            self.label = self.query_knn[:,:100]
            print("# loading time : {}s".format(time.time()- start_time))
            print("# train dist : {}".format(self.train_knn.shape))
            print("# query dist : {}".format(self.query_knn.shape))
            print("# valid dist : {}".format(self.valid_knn.shape))

    def load_sub(self):
        sub_dir = "folder/{}/{}/sub/".format(self.dataset, self.ntrain)
        os.makedirs(sub_dir, exist_ok= True)
        if not os.path.isfile(sub_dir + "train_sub.txt"):
            self.generate_sub()
            np.savetxt(sub_dir + "train_sub.txt", self.sub_strings, fmt='%s')
        else:
            print('# loading sub-strings from file')
            with open(sub_dir + "train_sub.txt", 'r') as file:
                self.sub_strings = [line.strip() for line in file.readlines()]
        self.ntrain = len(self.sub_strings)
        if not os.path.isfile(sub_dir + "train_sub_dist.txt"):
            self.train_sub_dist, self.train_sub_normalized_dist, self.train_sub_knn = get_dist_knn(
                self.beta, self.sub_strings
            )
            np.savetxt(sub_dir + "train_sub_dist.txt", self.train_sub_dist, fmt='%d')
            np.savetxt(sub_dir + "train_sub_normalized_dist.txt", self.train_sub_normalized_dist)
            np.savetxt(sub_dir + "train_sub_knn.txt", self.train_sub_knn, fmt='%d')
        else:
            self.train_sub_dist = np.loadtxt(sub_dir + "train_sub_dist.txt",dtype= int)
            self.train_sub_knn = np.loadtxt(sub_dir + "train_sub_knn.txt",dtype= int)
            self.train_sub_normalized_dist = np.loadtxt(sub_dir + "train_sub_normalized_dist.txt")
            print("# train sub dist : {}".format(self.train_sub_knn.shape))


    """
    有多种采子串的方法
    """
    # def generate_sub(self):
    #     """
    #     random起始位置 取100个字符
    #     """
    #     sub_length = 100
    #     strings = [self.data[i] for i in self.train_ids]
    #     sub_strings = set()
    #     for string in strings:
    #         sub_strings.add(string)
    #         seq_len = len(string)
    #         if seq_len < sub_length:
    #             continue
    #         start_pos = random.randint(0, seq_len - sub_length)
    #         subsequence = string[start_pos: start_pos + sub_length]
    #         sub_strings.add(subsequence)
    #     self.sub_strings = list(sub_strings)

    # def generate_sub(self):
        # """
        # 按照一定的间隔取100
        # """
    #     strings = [self.data[i] for i in self.train_ids]
    #     # sub_strings = []
    #     sub_strings = set()
    #     for i in range(self.ntrain):
    #         string = strings[i]
    #         length = len(string)
    #         # sub_strings.append(string)
    #         sub_strings.add(string)
    #         for j in range(step, length, step):
    #             # sub_strings.append(string[:j])
    #             sub_strings.add(string[:j])
    #     self.sub_strings = list(sub_strings)

    # def generate_sub(self):
        # """
        # 蛋白质序列根据特殊的字符取当中的序列(uniref_300, uniref_500)
        # """
    #     start_code = "ATG"
    #     stop_code = ["TGA","TAG","TAA"]

    #     strings = [self.data[i] for i in self.train_ids]
    #     sub_strings = set()
    #     for string in strings:
    #         sub_strings.add(string)
    #         start_index = string.find(start_code)
    #         if start_index == -1:
    #             continue
    #         stop_index = -1
    #         for code in stop_code:
    #             code_index = string.find(code, start_index + 3)
    #             if code_index != -1:
    #                 stop_index = code_index
    #                 break

    #         if stop_index == -1:
    #             continue
    #         sub_string = string[start_index:stop_index + 3]
    #         sub_strings.add(sub_string)
    #     self.sub_strings = list(sub_strings)

    def generate_sub(self):
        """
        按照空格划分子串(uniref_name)
        """
        strings = [self.data[i] for i in self.train_ids]
        sub_strings = set()  # 使用集合来去重
        for string in strings:
            words = string.split()  # 按空格拆分成单词
            for i in range(1, len(words) + 1):
                sub_string = ' '.join(words[:i])  # 构建子字符串
                sub_strings.add(sub_string)  # 添加子字符串
        self.sub_strings = list(sub_strings)  # 将集合转换回列表

    # def generate_sub(self):
        # """
        # 滑动窗口
        # """
    #     strings = [self.data[i] for i in self.train_ids]
    #     sub_strings = set()
    #     for string in strings:
    #         length = len(string)
    #         sub_strings.add(string)
    #         possible_substrings = [string[start:start + window_size] for start in range(0, length - window_size + 1, step)]
    #         # 从可能的子字符串中随机选择num_samples个
    #         # num_substrings_to_add = min(num_samples, len(possible_substrings))
    #         # chosen_substrings = random.sample(possible_substrings, num_substrings_to_add)
    #         sub_strings.update(possible_substrings)
    #     self.sub_strings = list(sub_strings)

    def save_split(self):
        root_dir = 'folder/{}/'.format(self.dataset)
        os.makedirs(root_dir, exist_ok=True)
        with open(root_dir + "test", "w") as w:
            w.writelines("%s\n" % index for index in self.test_indices)

    def build_vocab(self):
        characters = set()
        for sentence in self.data:
            characters.update(set(sentence))
        sorted_vocab = ['<pad>'] + ['<CLS>'] + ['<SEP>'] + sorted(characters)
        self.vocab = {char: index for index, char in enumerate(sorted_vocab)}

    def pad_data(self):
        valid_data = [self.data[idx] for idx in self.valid_ids]
        valid_tokens = self.tokenizer.batch_tokenize(valid_data)
        self.valid_sequence = self.tokenizer.convert_tokens_to_ids(valid_tokens)
    
    # def save_split(self):
    #     # 定义要保存的文件路径
    #     saved_path = "preprocess/{}/sampled_indices.txt".format(self.dataset)
    #     # 打开文件以写入模式
    #     with open(saved_path, "w") as file:
    #         # 遍历每一行
    #         for row in self.test_indices:
    #             file.write(row + "\n")

    def data_generator(self):
        indices = self.train_ids.copy()
        random.shuffle(indices)
        for i in range(0, self.ntrain, batch_size):
            # 打乱索引
            batch_indices = indices[i:i + batch_size]
            shuffled_indices = random.sample(self.train_ids, self.ntrain)
            shuffled_batch_indices = shuffled_indices[i:i + batch_size]
            anchor_batch = [self.data[idx] for idx in batch_indices]
            random_batch = [self.data[idx] for idx in shuffled_batch_indices]
            normalized_distance_batch = [self.normalized_distance[idx, random_idx] for idx, random_idx in zip(anchor_batch, random_batch)]
            distance_batch = torch.stack(normalized_distance_batch, dim = 0)
            #把句子转换成嵌入
            pad_anchor_sequence = char_to_pad(anchor_batch, self.vocab, self.maxlength, device)
            pad_random_sequence = char_to_pad(random_batch, self.vocab, self.maxlength, device)

            yield (pad_anchor_sequence, pad_random_sequence, distance_batch)
            
    def pos_neg_sample_data_generator(self, epoch):
        indices = self.train_ids.copy()
        # random.shuffle(indices)
        j = 0
        episode = self.epochs // 3
        while j < self.ntrain:
            anchor_input, pos_input, neg_input, pos_distance, neg_distance,pos_neg_distance = [],[],[],[],[],[]
            # start_time = time.time()
            for i in range(batch_size):
                if (i + j) < self.ntrain:
                    if self.sub:
                        if self.twosample:
                            if epoch <= episode:
                                pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_distance_sampling_1000_half(self.train_sub_knn, self.ntrain, i + j)
                            else:
                                pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_distance_sampling_knn_half(self.train_sub_knn, self.ntrain, i + j)
                        else:
                            pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_bio_kNN(self.train_sub_knn, self.ntrain, i + j)
                    else:
                        if self.twosample:
                            if epoch <= episode:
                                pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_CNNED(self.train_knn, self.ntrain, i + j)
                            else:
                                pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_distance_sampling_knn_half(self.train_knn, self.ntrain, i + j)
                        else:
                            pos_sampling_index_list, neg_sampling_index_list = sm.pos_neg_bio_kNN(self.train_knn, self.ntrain, i + j)
                        
                    for index in range(sampling_num):
                        pos_index = pos_sampling_index_list[index]
                        neg_index = neg_sampling_index_list[index]
                        if self.sub:
                            anchor_input.append(self.sub_strings[j + i])
                            pos_input.append(self.sub_strings[pos_index])
                            neg_input.append(self.sub_strings[neg_index])
                            pos_distance.append(self.train_sub_normalized_dist[j + i, pos_index])
                            neg_distance.append(self.train_sub_normalized_dist[j + i, neg_index])
                            pos_neg_distance.append(self.train_sub_normalized_dist[pos_index, neg_index])
                        else:
                            anchor_input.append(self.data[indices[j + i]])
                            pos_input.append(self.data[indices[pos_index]])
                            neg_input.append(self.data[indices[neg_index]])
                            pos_distance.append(self.train_normalized_dist[j + i, pos_index])
                            neg_distance.append(self.train_normalized_dist[j + i, neg_index])
                            pos_neg_distance.append(self.train_normalized_dist[pos_index, neg_index])
                else:
                    break
            anchor_tokens = self.tokenizer.batch_tokenize(anchor_input)
            anchor_sequence = self.tokenizer.convert_tokens_to_ids(anchor_tokens)
            pos_tokens = self.tokenizer.batch_tokenize(pos_input)
            pos_sequence = self.tokenizer.convert_tokens_to_ids(pos_tokens)
            neg_tokens = self.tokenizer.batch_tokenize(neg_input)
            neg_sequence = self.tokenizer.convert_tokens_to_ids(neg_tokens)
            pos_distance = torch.tensor(pos_distance).to(device)
            neg_distance = torch.tensor(neg_distance).to(device)
            pos_neg_distance = torch.tensor(pos_neg_distance).to(device)               
            yield(anchor_sequence, pos_sequence, neg_sequence, pos_distance, neg_distance, pos_neg_distance)
            j += batch_size


    def valid_data_generator(self):
        j = 0
        while j < self.nvalid:
            batch_indices = self.valid_ids[j:j + batch_size_valid]
            batch_input = [self.data[idx] for idx in batch_indices]
            distance_batch = []
            for i in range(batch_size_valid):
                if (i + j) < self.nvalid:
                    distance_batch.append(self.valid_normalized_dist[i + j])    
                stack_distance = np.vstack(distance_batch)
            batch_tokens = self.tokenizer.batch_tokenize(batch_input)
            batch_sequence = self.tokenizer.convert_tokens_to_ids(batch_tokens)
            distance_batch = torch.tensor(stack_distance).to(device)
            yield (batch_sequence, distance_batch)
            j += batch_size_valid

    def test_data_generator(self):
        self.base = [self.data[idx] for idx in self.base_ids]
        base_token = self.tokenizer.batch_tokenize(self.base)
        self.pad_base = self.tokenizer.convert_tokens_to_ids(base_token)
        j = 0
        while j < len(self.query_ids):
            batch_indices = self.query_ids[j:j + batch_size_valid]
            query_batch = [self.data[idx] for idx in batch_indices]
            distance_batch = []
            for i in range(batch_size_valid):
                if (i + j) < len(self.query_ids):
                    distance_batch.append(self.query_normalized_dist[i + j])
                stack_distance = np.vstack(distance_batch)
            batch_tokens = self.tokenizer.batch_tokenize(query_batch)
            batch_sequence = self.tokenizer.convert_tokens_to_ids(batch_tokens)
            distance_batch = torch.tensor(stack_distance).to(device)
            yield (batch_sequence, distance_batch)
            j += batch_size_valid

    def base_data_generator(self):
        j = 0
        while j < len(self.base_ids):
            batch_indices = self.base_ids[j:j + batch_size_valid]
            base_batch = [self.data[idx] for idx in batch_indices]
            base_token = self.tokenizer.batch_tokenize(base_batch)
            batch_sequence = self.tokenizer.convert_tokens_to_ids(base_token)
            yield (batch_sequence)
            j += batch_size_valid


class CharTokenizer:
    def __init__(self, vocab, max_length, cls_token='<CLS>', sep_token='<SEP>', pad_token='<pad>'):
        """
        初始化字符级别的Tokenizer

        :param cls_token: 句子的起始token
        :param sep_token: 句子的结束token
        """
        self.vocab = vocab
        self.max_length = max_length
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token

    def tokenize(self, text):
        """
        将输入文本分割为字符并添加特殊token

        :param text: 输入字符串
        :return: 带有特殊token的字符列表
        """

        tokens = list(text)
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        return tokens
    
    def batch_tokenize(self, texts):
        return [self.tokenize(text) for text in texts]

    def token_to_id(self, text):
        return [self.vocab[char] for char in text]

    def convert_tokens_to_ids(self, tokens):
        sequence = [self.token_to_id(sentence) for sentence in tokens]
        sequence = [torch.tensor(seq) for seq in sequence]
        return sequence

    def padsequence(self, sequences):
        padded_sequences = [seq[:self.max_length] if len(seq) > self.max_length else torch.cat([seq, torch.zeros(self.max_length - len(seq), dtype = torch.int, device=seq.device)]) for seq in sequences]
        return pad_sequence(padded_sequences, batch_first=True, padding_value=0).to(device)


class CharOneHotTokenizer:
    def __init__(self, vocab, max_length, cls_token='<CLS>', sep_token='<SEP>', pad_token='<pad>'):
        """
        初始化字符级别的Tokenizer

        :param cls_token: 句子的起始token
        :param sep_token: 句子的结束token
        """
        self.vocab = vocab
        self.max_length = max_length
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.vocab_size = len(vocab)

    def tokenize(self, text):
        """
        将输入文本分割为字符并添加特殊token
        :param text: 输入字符串
        :return: 带有特殊token的字符列表
        """
        # 将输入文本拆分为字符
        tokens = list(text)
        # 在开头添加 [CLS] token，在结尾添加 [SEP] token
        tokens = [self.cls_token] + tokens + [self.sep_token]
        return tokens
    
    def batch_tokenize(self, texts):
        """
        把一整个batch的字符串都按照字符切分并添加特殊token
        """
        return [self.tokenize(text) for text in texts]

    def token_to_id(self, text):
        """
        把一个字符串换成索引
        """
        return [self.vocab[char] for char in text]
    
    def index_to_one_hot(self, x):
        # add one row of zeros because the -1 represents the absence of element and it is encoded with zeros
        x = torch.cat((torch.eye(self.vocab_size, device=device), torch.zeros((1, self.vocab_size), device=device)), dim=0)[x]
        return x
            
    def convert_tokens_to_ids(self, tokens):
        sequence = [self.token_to_id(sentence) for sentence in tokens]
        pad_sequence = padsequence(sequence, self.max_length, device)
        one_hot_dic = [self.index_to_one_hot(seq) for seq in pad_sequence]
        return one_hot_dic

    
    
