import time
import string
import os
import re
import numpy as np
import random
import pickle
import torch
import argparse

from conf import device
from Bio import SeqIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count 
from dataset import CharTokenizer, Dataset


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_txt_line(file_path):
    n = 200000
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= n:
                break
            lines.append(line.strip())
    return lines

def write_txt(file_path, data):
    with open(file_path,"w") as file:
        for line in data:
            file.write(str(line) + '\n')

def random_sentences(file_path, points, nums):
    # 取points个点，连续取nums条
    selected_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        num_lines = len(lines)
        for _ in range(points):
            start_index = random.randint(0, num_lines - nums)
            for i in range(nums):
                selected_sentences.append(lines[start_index + i].strip())
    return selected_sentences

def random_title(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    selected_title = random.sample(lines, 30000)
    return selected_title

def remove_non_english_punctuation(english_sentence, german_sentence):
    # 获取英文中包含的标点符号
    punctuation = set(string.punctuation)

    english_punctuation = set()

    for char in english_sentence:
        if char in punctuation:
            english_punctuation.add(char)

    # 保留德语中与英文相同的标点符号
    german_sentence_cleaned = ''.join(char if char.isalpha() or char.isspace() or char in english_punctuation else '' for char in german_sentence)
    return german_sentence_cleaned

def remove_punctuation(source_sentences,german_sentences):
    cleaned_german_sentences = []
    for source_sentence,german_sentence in zip(source_sentences,german_sentences):
        removed_sentence = remove_non_english_punctuation(source_sentence,german_sentence)
        #数据集在标点符号左右两个都有一个空格，去掉连续两个空格中的一个
        removed_sentence = re.sub(r' +', ' ', removed_sentence)
        cleaned_german_sentences.append(removed_sentence)
    
    return cleaned_german_sentences  

def sort_sentences(sentences):
    return sorted(sentences)


def process_line(line):
    return line[:500], line[:100], line[:50], line[:200], line[:300]


def process_500_300_200_100_50(source_path):
    lines = read_txt(source_path)
    sequence_500 = []
    sequence_300 = []
    sequence_200 = []
    sequence_100 = []
    sequence_50 = []

    for line in lines:
        line_500, line_100, line_50, line_200, line_300 = process_line(line)
        sequence_500.append(line_500)
        sequence_300.append(line_300)
        sequence_200.append(line_200)
        sequence_100.append(line_100)
        sequence_50.append(line_50)
    return sequence_500, sequence_300, sequence_200, sequence_100, sequence_50

def analyze_sentences(file_path):
    min_length = float('inf')
    max_length = 0
    total_length = 0
    count = 0

    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            length = len(sentence)
            if length < min_length:
                min_length = length
            if length > max_length:
                max_length = length
            total_length += length
            count += 1
        
    average_length = total_length / count
    print("min:", min_length)
    print("max:", max_length)
    print("average:", average_length)
    print("item:", count)
    return min_length, max_length, average_length, count

def preprocess_data(dataset_name):
    folder_path = 'preprocess/{}'.format(dataset_name)
    if not os.path.exists(folder_path):
        # 使用 os.makedirs() 创建文件夹
        os.makedirs(folder_path)

    if dataset_name == 'WMT':
        source_path = '.data/WMT/train_src.en'
        mt_path = '.data/WMT/train_mt.de'
        pe_path = '.data/WMT/train_pe.de'
        #按行读取两个文件
        source_sentences = read_txt(source_path)
        mt_sentences = read_txt(mt_path)
        pe_sentences = read_txt(pe_path)
        #去除掉数据中多余的标点符号
        cleaned_mt_sentences = remove_punctuation(source_sentences,mt_sentences)
        cleaned_pe_sentences = remove_punctuation(source_sentences,pe_sentences)
        write_txt('preprocess/WMT/mt_de.txt',cleaned_mt_sentences)
        write_txt('preprocess/WMT/pe_de.txt',cleaned_pe_sentences)

    elif dataset_name == 'word':
        source_path = '.data/word/word'
        # 随机选择10000个单词，并且按照顺序连续取5个，这样总共能随机取到50000条数据
        selected_sentence = random_sentences(source_path)
        selected_sentence_path =  'preprocess/{}/word.txt'.format(dataset_name)
        write_txt(selected_sentence_path, selected_sentence)

    elif dataset_name =='DBLP':
        source_path = '.data/DBLP/title.txt'
        # 随机选择30000条标题作为数据
        selected_title = random_title(source_path)
        selected_title_path = 'preprocess/{}/title.txt'.format(dataset_name)
        write_txt(selected_title_path, selected_title)

    elif dataset_name == 'uniref':
        # uniref包含蛋白质序列和蛋白质名字两个数据
        # 这个文件里包含50000条蛋白质名字
        name_path = '.data/uniref/uniref50/protein_name.txt'
        # 这个文件里包含100000条蛋白质序列
        sequence_path = '.data/uniref/uniref50/protein_sequences.txt'
        # sequence_path = '.data/uniref/uniref90/protein_sequences_article.txt'
        # 把序列按照50，100，200，300，500的长度，划分成多个数据集
        sequence_500_path = 'preprocess/uniref/protein_sequences_500.txt'
        sequence_300_path = 'preprocess/uniref/protein_sequences_300.txt'
        sequence_200_path = 'preprocess/uniref/protein_sequences_200.txt'
        sequence_100_path = 'preprocess/uniref/protein_sequences_100.txt'
        sequence_50_path = 'preprocess/uniref/protein_sequences_50.txt'
        # sequence_path = 'preprocess/uniref/sorted_protein_sequences_90.txt'
        sorted_sequence_path = 'preprocess/{}/sorted_protein_sequences_90.txt'.format(dataset_name)
        sorted_names_path = 'preprocess/{}/sorted_protein_names.txt'.format(dataset_name)
        # sorted_sequence_path = 'preprocess/{}/sorted_protein_sequences_90_20w.txt'.format(dataset_name)
        # # protein_names = read_txt(source_path)
        # protein_sequences = read_txt_line(sequence_path)
        # 读取序列，并排序
        protein_sequences = read_txt(sequence_path)
        protein_names = read_txt(name_path)
        sorted_protein_sequence = sort_sentences(protein_sequences)
        sorted_protein_name = sort_sentences(protein_names)
        write_txt(sorted_names_path, sorted_protein_name)
        write_txt(sorted_sequence_path, sorted_protein_sequence)
        selected_protein_names = random_sentences(sorted_names_path)
        selected_protein_sequences = random_sentences(sorted_sequence_path)
        selected_protein_names_path = 'preprocess/{}/protein_names.txt'.format(dataset_name)
        selected_protein_sequences_path = 'preprocess/{}/protein_sequences_90.txt'.format(dataset_name)
        write_txt(selected_protein_sequences_path, selected_protein_sequences)
        write_txt(selected_protein_names_path, selected_protein_names)
        sequence_500, sequence_300, sequence_200, sequence_100, sequence_50 = process_500_300_200_100_50(selected_protein_sequences_path)
        sequence_500 = sort_sentences(sequence_500)
        sequence_300 = sort_sentences(sequence_300)
        sequence_200 = sort_sentences(sequence_200)
        sequence_100 = sort_sentences(sequence_100)
        sequence_50 = sort_sentences(sequence_50)
        write_txt(sequence_500_path, sequence_500)
        write_txt(sequence_300_path, sequence_300)
        write_txt(sequence_200_path, sequence_200)
        write_txt(sequence_100_path, sequence_100)
        write_txt(sequence_50_path, sequence_50)
        # analyze_sentences(sequence_path)

    elif dataset_name == 'gen50ks':
        sequence_path = '.data/Gen50ks/gen50ks.txt'
        # analyze_sentences(sequence_path)
        sequence_500_path = 'preprocess/{}/gen50ks_500.txt'.format(dataset_name)
        sequence_300_path = 'preprocess/{}/protein_sequences_300.txt'.format(dataset_name)
        sequence_200_path = 'preprocess/{}/protein_sequences_200.txt'.format(dataset_name)
        sequence_100_path = 'preprocess/{}/gen50ks_100.txt'.format(dataset_name)
        sequence_50_path = 'preprocess/{}/gen50ks_50.txt'.format(dataset_name)
        # source_sequence = read_txt(sequence_path)
        sequence_500, sequence_100, sequence_50 = process_500_300_200_100_50(sequence_path)
        sequence_500 = sort_sentences(sequence_500)
        sequence_300 = sort_sentences(sequence_300)
        sequence_200 = sort_sentences(sequence_200)
        sequence_100 = sort_sentences(sequence_100)
        sequence_50 = sort_sentences(sequence_50)
        write_txt(sequence_500_path, sequence_500)
        write_txt(sequence_300_path, sequence_300)
        write_txt(sequence_200_path, sequence_200)
        write_txt(sequence_100_path, sequence_100)
        write_txt(sequence_50_path, sequence_50)

    elif dataset_name == 'trec':
        sequence_path = '.data/Trec/trec.txt'
        analyze_sentences(sequence_path)

    elif dataset_name == 'uniprot':
        sequence_path = '.data/uniprot/uniprot_sprot.txt'
        sorted_sequence_path = 'preprocess/uniprot/sorted_sequence.txt'
        random_path = 'preprocess/{}/uniprot.txt'.format(dataset_name)
        sequence_500_path = 'preprocess/{}/uniprot_500.txt'.format(dataset_name)
        sequence_300_path = 'preprocess/{}/uniprot_300.txt'.format(dataset_name)
        sequence_200_path = 'preprocess/{}/uniprot_200.txt'.format(dataset_name)
        sequence_100_path = 'preprocess/{}/uniprot_100.txt'.format(dataset_name)
        sequence_50_path = 'preprocess/{}/uniprot_50.txt'.format(dataset_name)
        sequence = read_txt(sequence_path)
        sorted_sequence = sort_sentences(sequence)
        write_txt(sorted_sequence_path, sorted_sequence)
        random_sequence = random_sentences(sorted_sequence_path)
        write_txt(random_path, random_sequence)
        sequence_500, sequence_300, sequence_200, sequence_100, sequence_50 = process_500_300_200_100_50(random_path)
        write_txt(sequence_500_path, sequence_500)
        write_txt(sequence_300_path, sequence_300)
        write_txt(sequence_200_path, sequence_200)
        write_txt(sequence_100_path, sequence_100)
        write_txt(sequence_50_path, sequence_50)

def analyse_dist(dist_path):

    # dist_path = 'folder/{}/dis/train_dist.txt'.format(dataset_name)
    train_dist = np.loadtxt(dist_path, dtype=int)
    average_dist = np.mean(train_dist)
    max_dist = np.max(train_dist)
    median_dist = np.median(train_dist)
    print('max:', max_dist)
    print('ave:', average_dist)
    print('median:',median_dist)
    normalized_ave = average_dist / max_dist
    beta_ave = -np.log(0.5) / normalized_ave
    normalized_med = median_dist / max_dist
    beta_med = -np.log(0.5) / normalized_med
    print('beta_ave:', beta_ave)
    print('beta_med:', beta_med)

def sub_strings(sequences, n):
    sub_strings = []
    for i in range(len(sequences)):
        sub_strings.append(sequences[i][:n])
    return sub_strings

def read_fasta(file_path):
    total_lines = 0
    sequence_lengths = []

    # 读取FASTA文件并统计序列长度
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # 记录序列长度
            seq_len = len(record.seq)
            sequence_lengths.append(seq_len)

            # 计算序列行数（每个序列包含一个ID行和一行或多行的序列）
            total_lines += 1

    # 统计最大值、最小值和平均值
    max_len = max(sequence_lengths) if sequence_lengths else 0
    min_len = min(sequence_lengths) if sequence_lengths else 0
    avg_len = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0

    # 打印结果
    print(f"Total lines (number of sequences): {total_lines}")
    print(f"Max sequence length: {max_len}")
    print(f"Min sequence length: {min_len}")
    print(f"Average sequence length: {avg_len:.2f}")

def convert_fasta_to_txt(fasta_file, output_txt_file):
    with open(fasta_file, 'r') as f_in, open(output_txt_file, 'w') as f_out:
        sequence = ""
        for line in f_in:
            if line.startswith('>'):
                # If we encounter a new sequence header, write the current sequence to the file
                if sequence:
                    f_out.write(sequence + '\n')
                # f_out.write(line)  # Write the header line
                sequence = ""  # Reset the sequence for the next one
            else:
                sequence += line.strip()  # Concatenate the sequence parts (no internal newlines)
        # Write the last sequence after the loop ends
        if sequence:
            f_out.write(sequence + '\n')
            

def remove(sequences):
    return list(set(sequences))

def cut_sequences(sequences, max_length=300):
    """
    对所有序列按照长度300进行切割
    :param sequences: 序列列表
    :param max_length: 最大长度
    :return: 切割后的序列列表
    """
    cut_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            # 按300为单位切割长序列
            for i in range(0, len(seq), max_length):
                cut_sequences.append(seq[i:i+max_length])
        else:
            cut_sequences.append(seq)  # 保留原始序列
    return cut_sequences

def process_uniprot(file_path):
    sequences = read_txt(file_path)
    sequences_300 = cut_sequences(sequences)
    sequences_300 = remove(sequences_300)
    sorted_sequences_300 = sort_sentences(sequences_300)
    write_txt('preprocess/uniprot/sort_uniprot_300.txt', sorted_sequences_300)
    selected_sequences = random_sentences('preprocess/uniprot/sort_uniprot_300.txt')
    write_txt('preprocess/uniprot/uniprot_300.txt',selected_sequences)

def read_and_truncate(file_path, max_length):
    with open(file_path, 'r') as f:
        lines = [line.strip()[:max_length] for line in f]
    return lines


def generate_pkl(args):

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
    tokenizer = CharTokenizer(data_loader.vocab, data_loader.maxlength)
    if args.mode == 'train':
        save_path = 'preprocess/{}.pkl'.format(args.dataset)
        train = [data_loader.data[idx] for idx in data_loader.train_ids]
        valid = [data_loader.data[idx] for idx in data_loader.valid_ids]

        train_tokens = tokenizer.batch_tokenize(train)
        train_sequences = tokenizer.convert_tokens_to_ids(train_tokens)
        train_sequences = tokenizer.padsequence(train_sequences)

        valid_tokens = tokenizer.batch_tokenize(valid)
        valid_sequences = tokenizer.convert_tokens_to_ids(valid_tokens)
        valid_sequences = tokenizer.padsequence(valid_sequences)

        train_dist = torch.tensor(data_loader.train_dist, device=device)
        valid_dist = torch.tensor(data_loader.valid_dist, device=device)

        sequences = {
            'train': train_sequences,
            'val': valid_sequences
        }

        distances = {
            'train':train_dist,
            'val':valid_dist

        }
        with open(save_path, 'wb') as f:
            pickle.dump((sequences, distances),f)

    elif args.mode == 'test':
        save_path = 'preprocess/closest_{}.pkl'.format(args.dataset)

        base = [data_loader.data[idx] for idx in data_loader.base_ids]
        query = [data_loader.data[idx] for idx in data_loader.query_ids]

        base_tokens = tokenizer.batch_tokenize(base)
        base_sequences = tokenizer.convert_tokens_to_ids(base_tokens)
        sequences_references = tokenizer.padsequence(base_sequences)

        query_tokens = tokenizer.batch_tokenize(query)
        query_sequences = tokenizer.convert_tokens_to_ids(query_tokens)
        sequences_queries= tokenizer.padsequence(query_sequences) 

        # labels = torch.tensor(data_loader.label, device=device)
        normalized_dist = torch.tensor(data_loader.query_normalized_dist, device = device)

        with open(save_path, 'wb') as f:
            pickle.dump((sequences_references, sequences_queries, normalized_dist), f)


# def process_bioknn():
#     train_file_path = '.data/open_source_datasets/uniprot/train_seq_list.txt'
#     query_file_path = '.data/open_source_datasets/uniprot/query_seq_list.txt'
#     base_file_path = '.data/open_source_datasets/uniprot/base_seq_list.txt'
#     output_file_path = '.data/open_source_datasets/uniprot/uniprot_300.txt'
#     train_idx_path = 'folder/uniprot_300/1000/idx/train_idx.txt'
#     valid_idx_path = 'folder/uniprot_300/1000/idx/valid_idx.txt'
#     query_idx_path = 'folder/uniprot_300/1000/idx/query_idx.txt'
#     base_idx_path = 'folder/uniprot_300/1000/idx/base_idx.txt'
    
#     max_len = 300
#     # 读取并处理 train 和 query 文件
#     train_seqs = read_and_truncate(train_file_path, max_len)
#     query_seqs = read_and_truncate(query_file_path, max_len)

#     with open(output_file_path, 'w') as out_f, open(train_idx_path, 'w') as train_idx_f, open(query_idx_path, 'w') as query_idx_f:
#         for idx, seq in enumerate(train_seqs):
#             out_f.write(seq + '\n')
#             train_idx_f.write(f"{idx}\n")  # 记录train序列的索引
#         for idx, seq in enumerate(query_seqs, start=len(train_seqs)):
#             out_f.write(seq + '\n')
#             query_idx_f.write(f"{idx}\n")  # 记录query序列的索引

#         # 读取 base 文件中的所有序列
#     base_seqs = read_and_truncate(base_file_path, max_len)

#     # 随机选取1000条base序列，截取并记录索引
#     valid_1000_indices = random.sample(range(len(base_seqs)), 1000)
#     with open(output_file_path, 'a') as out_f, open(valid_idx_path, 'w') as base_idx_f:
#         for idx, base_idx in enumerate(valid_1000_indices, start=len(train_seqs) + len(query_seqs)):
#             out_f.write(base_seqs[base_idx] + '\n')
#             base_idx_f.write(f"{idx}\n")  # 记录base序列的索引

#     # 从剩余的 base_seqs 中排除已经选中的1000条，进行排序
#     remaining_indices = sorted(set(range(len(base_seqs))) - set(valid_1000_indices))
#     remaining_seqs = [base_seqs[i] for i in remaining_indices]

#     # 从排序后的剩余序列中，随机选取4700个起始点，每个连续取10条
#     base_indices = []
#     for _ in range(4700):
#         start_idx = random.choice(range(len(remaining_seqs) - 10))  # 确保不会越界
#         base_indices.extend(remaining_indices[start_idx:start_idx + 10])

#     # 写入 valid 序列到输出文件，并记录索引
#     with open(output_file_path, 'a') as out_f, open(base_idx_path, 'w') as valid_idx_f:
#         for idx, valid_idx in enumerate(base_indices, start=len(train_seqs) + len(query_seqs) + 1000):
#             out_f.write(base_seqs[valid_idx] + '\n')
#             valid_idx_f.write(f"{idx}\n")  # 记录valid序列的索引

def process_bioknn():
    base_file_path = '.data/open_source_datasets/uniprot/base_seq_list.txt'
    sorted_base_file_path = '.data/open_source_datasets/uniprot/sorted_base_seq_list.txt'
    uniprot_300_seq = '.data/open_source_datasets/uniprot/uniprot_300.txt'
    max_len = 300
    base_seqs = read_and_truncate(base_file_path, max_len)
    sort_base_seqs = sort_sentences(base_seqs)
    write_txt(sorted_base_file_path, sort_base_seqs)
    sequences = random_sentences(sorted_base_file_path, 10000, 5)
    write_txt(uniprot_300_seq, sequences)
    
if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    # # # 添加 dataset 参数
    parser.add_argument("--dataset", type=str, default="word", choices=["WMT", "word","DBLP","uniref_100","uniref_500","uniref_300","uniref_name","gen50ks_50","gen50ks_100","gen50ks_500","trec"], help="Name of the dataset")
    parser.add_argument("--epochs", type=int, default=90, help="# number of epochs")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nv", type=int, default=1000, help="# of valid items")
    parser.add_argument("--augment", action="store_true", default=False, help="data augmentation")
    parser.add_argument("--sub", action="store_true", default=False, help="sub-string")
    parser.add_argument("--twosample", action="store_true", default=False, help="twosample")
    parser.add_argument("--mode", type=str, default="train")
    # # # 解析命令行参数
    args = parser.parse_args()
    # read_fasta('/data/b34/uniref90.fasta')
    # preprocess_data(args.dataset)
    # analyse_dist('folder/DBLP/1000/dis/train_dist.txt')
    # read_fasta('/data/b34/uniref90.fasta')
    generate_pkl(args)
