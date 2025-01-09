
import matplotlib.pyplot as plt
import argparse
import re
import numpy as np
from conf import sampling_num


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode, args):
    if mode == 'loss':
        train = read('result/{}/train_loss.txt'.format(args.dataset))
        valid = read('result/{}/valid_loss.txt'.format(args.dataset))
        # test = read('result/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(valid, 'b', label='validation')
        # plt.plot(test, 'g', label='test')
        plt.legend(loc='lower left')


    elif mode == 'bleu':
        bleu = read('./result/bleu.txt')
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig('result/{}-{}-train_result.jpg'.format(args.dataset, sampling_num))
    plt.show()
    

def draw_loss(mode):
    if mode == 'loss':
        train_path = 'result/gen50ks_100/train_loss_CNN.txt'
        valid_path = 'result/gen50ks_100/valid_loss_CNN.txt'
        train = read(train_path)
        valid = read(valid_path)
        # test = read('result/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(valid, 'b', label='validation')
        # plt.plot(test, 'g', label='test')
        plt.legend(loc='lower left')


    # elif mode == 'bleu':
    #     bleu = read('./result/bleu.txt')
    #     plt.plot(bleu, 'b', label='bleu score')
    #     plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig('result/train_result.jpg')
    plt.show()

def data_distribution(data):
    
    data_flat = data.flatten()

    # 绘制直方图
    plt.hist(data_flat, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('result/data_distribution.jpg')
    plt.show()

def draw_recall_1():
    methods = ['CGK', 'GRU', 'Glocal T.', 'Local T.', 'CNNED', 'Char T.']
    items_k = np.array([1, 10, 100, 1000, 10000])  # x轴（对数）
    recall_data = {
        'AsMac': [50, 55, 60, 65, 70],
        'GRU': [55, 60, 65, 70, 75],
        'Glocal T.': [60, 65, 70, 75, 80],
        'Local T.': [58, 63, 68, 73, 78],
        'CNNED': [65, 75, 85, 90, 95],
        'Bio-kNN': [60, 70, 80, 85, 90]
    }

    # 开始绘图
    plt.figure(figsize=(10, 5))

    for method in methods:
        plt.plot(items_k, recall_data[method], label=method, marker='o')

    plt.xscale('log')  # 设置x轴为对数
    plt.xlabel('# Items [k]')
    plt.ylabel('Top-1 Recall (%)')
    plt.title('Top-1 Recall curves for multiple methods')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='HyperParameters for CharTransformer')
    parser.add_argument("--dataset", type=str, default="word", help="dataset")
   
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    draw(mode='loss', args = args)
    # distance_data = np.loadtxt('preprocess/pe_distance.txt')
    # data_distribution(distance_data)
    # draw(mode='bleu')
