import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np
import torch.nn as nn

from conf import *
# from torch.nn import Module, Parameter

class WeightMSELoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            # self.weight.append(0.)
            for index in range(sampling_num):
                self.weight.append(float(sampling_num) - index)

        self.weight = torch.tensor(self.weight).to(device)
        self.weight /= torch.sum(self.weight)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        # self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad = False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU = False):
        div = target - input
        if isReLU:
            div = F.relu(div)

        ## 当最后一个batch没有被整除，这里会报错。
        square = torch.mul(div, div)
        weight_square = torch.mul(square, self.weight)

        loss = torch.sum(weight_square)
        return loss

class WeightedRankingLoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        mse_loss = self.positive_loss(p_input, p_target, False)
        ## to do：compare True,False
        negative_mse_loss = self.negative_loss(n_input, n_target, True)
        self.mse_loss = mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = mse_loss + negative_mse_loss
        return loss
        # mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False)
        # negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), True)
        # self.mse_loss = mse_loss
        # self.negative_mse_loss = negative_mse_loss
        # loss = sum([mse_loss,negative_mse_loss])
        # return loss

class TripletLoss(nn.Module):
    def __init__(self, epochs):
        super(TripletLoss, self).__init__()
        # 根据训练的epoch的数量讲损失参数进行了调整，Ls定义了训练不同阶段的损失参数（？为什么不同epoch会有不同的值）
        self.l, self.r = 1, 1
        self.criterion = nn.MSELoss().float()
        # step = epochs // 5
        # step = epochs // 3
        step = epochs
        
        self.Ls = {
            step * 0: (0, 10),
            # step * 1: (10, 10),
            # step * 2: (10, 1),
            # step * 3: (5, 0.1),
            # step * 4: (1, 0.01),
        }

        # self.Ls = {
            # step * 0: (0, 10),
            # step * 1: (10, 10),
            # step * 2: (10, 1),
        # }

        # self.Ls = {
        #     step * 0: (0, 10),
        # }

    def dist(self, ins, pos):
        # 计算欧氏距离
        return torch.norm(ins - pos, dim=1)

    # def forward(self, x, dists, epoch):
    #     if epoch in self.Ls:
    #         self.l, self.r = self.Ls[epoch]
    #     # 从x中分离出anchor，positive，negative的output
    #     anchor, positive, negative = x
    #     # 得到这些字符串之间的真实distance
    #     pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

    #     # 计算两两output之间的欧氏距离
    #     pos_embed_dist = self.dist(anchor, positive)
    #     neg_embed_dist = self.dist(anchor, negative)
    #     pos_neg_embed_dist = self.dist(positive, negative)

    #     threshold = neg_dist - pos_dist
    #     print('threshold:',threshold)
    #     # 计算排名损失
    #     print('pos:',pos_embed_dist)
    #     print('neg:',neg_embed_dist)
    #     rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

    #     print('rank', rank_loss)
    #     # MSELoss
    #     mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
    #                (neg_embed_dist - neg_dist) ** 2 + \
    #                (pos_neg_embed_dist - pos_neg_dist) ** 2
    #     print('mse',mse_loss)

    #     return torch.mean(rank_loss), \
    #            torch.mean(mse_loss), \
    #            torch.mean(self.l * rank_loss +
    #                       self.r *  torch.sqrt(mse_loss))

    def forward(self, pos_output, pos_distance, neg_output, neg_distance, pos_neg_output, pos_neg_distance, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        # self.l = 
        # output都是经过欧氏距离之后，再torch.exp
        # distance也是经过torch.exp，所以也是值越大表示越接近。
        # 所以这里就直接表示相似度，越大表示越相似

        threshold = pos_distance - neg_distance
        # threshold = neg_distance - pos_distance
        # print('threshold:',threshold)
        # 计算排名损失
        ## 增大pos_output - neg_output之间差值
        # rank_loss = F.relu(pos_output - neg_output + threshold)
        # 现在的值都是越大表示越相似，值越小表示越不相似
        # 正样本的相似度大于负样本的相似度，且差值大于'threshold'，损失为0，表示正确区分正负样本
        # 只有当没有正确区分，就会产生正的损失值
        # print('neg:',neg_output)
        # print('pos:',pos_output)
        
        rank_loss = F.relu(neg_output - pos_output + threshold)
        # rank_loss = F.relu(pos_output - neg_output + threshold)
        # print('rank:', rank_loss)
        # MSELoss

        # if torch.isnan(pos_diff).any():
        #     print(pos_diff)
        #     print("NaN detected in pos_diff calculation")
        # if torch.isnan(neg_diff).any():
        #     print(neg_diff)
        #     print("NaN detected in neg_diff calculation")
        # if torch.isnan(squared_diff).any():
        #     print(pos_neg_diff)
        #     print("NaN detected in pos_neg_diff calculation")

        # mse_loss = (pos_output.float() - pos_distance.float()) ** 2 + \
        #             (neg_output.float() - neg_distance.float()) ** 2 + \
        #             (pos_neg_output.float() - pos_neg_distance.float()) ** 2
        
        mse_loss = self.criterion(pos_output.float(), pos_distance.float()) + \
                    self.criterion(neg_output.float(), neg_distance.float()) + \
                    self.criterion(pos_neg_output.float(), pos_neg_distance.float())
        # print('mse_loss:', torch.sqrt(mse_loss))
        return  torch.mean(rank_loss), \
                torch.mean(mse_loss), \
                torch.mean(self.l * rank_loss +
                            self.r * torch.sqrt(mse_loss))