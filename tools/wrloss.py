import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np
import torch.nn as nn

from conf import *

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
        negative_mse_loss = self.negative_loss(n_input, n_target, True)
        self.mse_loss = mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = mse_loss + negative_mse_loss
        return loss

class TripletLoss(nn.Module):
    def __init__(self, epochs):
        super(TripletLoss, self).__init__()
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


    def dist(self, ins, pos):
        return torch.norm(ins - pos, dim=1)

  
    def forward(self, pos_output, pos_distance, neg_output, neg_distance, pos_neg_output, pos_neg_distance, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]

        threshold = pos_distance - neg_distance
        
        rank_loss = F.relu(neg_output - pos_output + threshold)
        
        mse_loss = self.criterion(pos_output.float(), pos_distance.float()) + \
                    self.criterion(neg_output.float(), neg_distance.float()) + \
                    self.criterion(pos_neg_output.float(), pos_neg_distance.float())
        return  torch.mean(rank_loss), \
                torch.mean(mse_loss), \
                torch.mean(self.l * rank_loss +
                            self.r * torch.sqrt(mse_loss))
