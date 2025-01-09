

import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import *
from utils import euclidean_distance, euclidean_distance_squared, poincare_distance, extract_non_padding_values, padsequence
from block import EncoderLayer, SEAttentionModule, SETensorNetworkModule, ST_Layer
from embedding import TransformerEmbedding, RelativePositionalEmbedding, OneHotEmbedding, AbsolutePositionalEncoding


POOL = nn.AvgPool1d

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_head, d_model, ffn_hidden, drop_prob, device ):
        super(Encoder,self).__init__()
        self.emb = TransformerEmbedding(d_model = d_model,
                                        max_len = max_len,
                                        vocab_size = vocab_size,
                                        drop_prob = drop_prob,
                                        device = device)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                  n_head = n_head,
                                                  drop_prob =drop_prob)
                                     for _ in range(n_layers)])
        
            
    def forward(self, enc_input, src_mask):
        enc_input = self.emb(enc_input)
        for layer in self.layers:
            enc_input = layer(enc_input, src_mask)
        return enc_input

class BiEncoder(nn.Module):
    def __init__(self, src_pad_idx, d_model, n_head, max_len, ffn_hidden, vocab_size, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(d_model= d_model,
                               n_head= n_head,
                               max_len= max_len,
                               ffn_hidden= ffn_hidden,
                               vocab_size= vocab_size,
                               n_layers= n_layers,
                               drop_prob= drop_prob,
                               device= device)
        # self.linear = Linear(d_model=d_model, dropout= drop_prob,device = device)
        self.linear = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)
        
    def forward(self, src_input, tar_input):
        src_mask = self.make_src_mask(src_input)
        tar_mask = self.make_src_mask(tar_input)
        enc_src = self.encoder(src_input, src_mask)
        enc_tar = self.encoder(tar_input, tar_mask)
        
        exp_src_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, d_model)
        exp_tar_mask = tar_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, d_model)

        src_output = self.linear(torch.mean(exp_src_mask * enc_src, dim = 1))
        tar_output = self.linear(torch.mean(exp_tar_mask * enc_tar, dim = 1))

        return src_output, tar_output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
class Linear(nn.Module):
    def __init__(self, dropout, device, d_model):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.device = device
        self.lin = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.lin.weight.data)
        nn.init.zeros_(self.lin.bias.data)
    
    def forward(self, scores):
        scores = F.relu(self.lin(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        return scores

class MLPModule(nn.Module):
    def __init__(self, nhid, dropout, d_model):
        super(MLPModule, self).__init__()
        self.nhid = nhid
        self.dropout = dropout
        self.device = device
        self.lin0 = nn.Linear(d_model, d_model / 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = nn.Linear(d_model / 2 , 1)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        nn.init.zeros_(self.lin1.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sigmoid(self.lin1(scores)).squeeze(-1)
        return scores
      
class Cross_Bi_Encoder(torch.nn.Module):


    def __init__(self, src_pad_idx, vocab_size, max_len, distance, scaling, mask):
        super(Cross_Bi_Encoder, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.distance = distance
        self.mask = mask
        self.scaling = None
        if scaling:
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            self.scaling = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.l = l
        self.r = r
        self.setup_layers()

    def normalize_embeddings(self, embeddings):

        """ Project embeddings to an hypersphere of a certain radius """
        min_scale = 1e-7
        if self.distance == 'hyperbolic':
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return F.normalize(embeddings, p=2, dim=1) * self.radius.clamp_min(min_scale).clamp_max(max_scale)

    def setup_layers(self):
        self.emb = TransformerEmbedding(d_model = d_model,
                                        max_len = self.max_len,
                                        vocab_size = self.vocab_size,
                                        drop_prob = drop_prob,
                                        device = device)
        self.biencoderlayer = EncoderLayer(d_model = d_model,
                                        ffn_hidden = ffn_hidden,
                                        n_head = n_head,
                                        drop_prob =drop_prob)
        
        self.rel_pos_emb = RelativePositionalEmbedding(max_len= 2 * self.max_len,
                                                   d_model= d_model)
        
        self.crossencoderlayer = EncoderLayer(d_model = d_model,
                                        ffn_hidden = ffn_hidden,
                                        n_head = n_head,
                                        drop_prob = drop_prob)
        
        self.ST_layer1 = ST_Layer(embedding_size=d_model,
                                  device= device)

        self.linear1 = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)

        self.ST_layer2 = ST_Layer(embedding_size=d_model,
                                  device= device)

        self.linear2 = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, 1)
        
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.zeros_(self.linear2.bias.data)

    def compute_distance(self, src_output, tar_output, beta):
        if self.distance == 'hyperbolic':
            if self.scaling is not None:
                src_output = self.normalize_embeddings(src_output)
                tar_output = self.normalize_embeddings(tar_output)
            dist = poincare_distance(src_output, tar_output)

            if self.scaling is not None:
                dist = dist * self.scaling
        elif self.distance == 'square':
            dist = euclidean_distance_squared(src_output, tar_output)
        elif self.distance == 'euclidean':
            dist = euclidean_distance(src_output, tar_output)
        return torch.exp(-beta * dist)

    def compute_lengths(self, sequence):
        lengths = [seq.size(0) for seq in sequence]
        return lengths
    
    def padseq(self, sequences, max_length):
        padded_sequence = [torch.cat([seq, torch.zeros(max_length - seq.size(0), d_model, dtype=seq.dtype, device = seq.device)]) for seq in sequences]
        return torch.stack(padded_sequence).to(device)

    def src_pad_tar_pad(self, src_input, tar_input, src_enc_output, tar_enc_output):
        original_pad_input = torch.cat([src_input, tar_input], dim = 1)
        ce_pad_input = torch.cat([src_enc_output, tar_enc_output], dim = 1)
        return original_pad_input,ce_pad_input

    def src_tar_pad(self, src_input, tar_input, src_lengths, tar_lengths, src_enc_output, tar_enc_output):
        combined_original_input = [torch.cat([t1, t2]) for t1,t2 in zip(src_input, tar_input)]
        original_pad_input = padsequence(combined_original_input, self.max_len * 2, device)
        combined_input = []
        for i in range(len(src_input)):
            len1 = src_lengths[i]
            len2 = tar_lengths[i]
            src = src_enc_output[i,:len1,:]
            tar = tar_enc_output[i,:len2,:]
            ce_input = torch.cat([src, tar], dim=0).to(device)
            combined_input.append(ce_input)
        ce_pad_input = self.padseq(combined_input, self.max_len * 2)
        return original_pad_input, ce_pad_input

    def forward(self, src_input, tar_input, beta):
        # start_time = time.time()
        src_pad_input = padsequence(src_input, self.max_len, device)
        tar_pad_input = padsequence(tar_input, self.max_len, device)

        src_lengths = self.compute_lengths(src_input)
        tar_lengths = self.compute_lengths(tar_input)

        src_mask = self.make_src_mask(src_pad_input)
        tar_mask = self.make_src_mask(tar_pad_input)

        exp_src_mask = src_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, d_model)
        exp_tar_mask = tar_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, d_model)

        src_embed = self.emb(src_pad_input)
        tar_embed = self.emb(tar_pad_input)

        src_enc_output = self.biencoderlayer(src_embed, src_mask)
        tar_enc_output = self.biencoderlayer(tar_embed, tar_mask)


        be_src_output = self.ST_layer1(src_enc_output, src_mask)
        be_tar_output = self.ST_layer1(tar_enc_output, tar_mask)

        
        bi_output = self.compute_distance(be_src_output, be_tar_output, beta)

        original_pad_input, ce_pad_input = self.src_pad_tar_pad(src_pad_input, tar_pad_input, src_enc_output, tar_enc_output)

        combined_global_mask, combined_cross_mask, crombined_local_mask = self.generate_local_cross_mask(original_pad_input, src_lengths, tar_lengths)

        exp_combined_mask = combined_global_mask.squeeze(1).squeeze(1).unsqueeze(2).expand(-1, -1, d_model)

        
        if self.mask == 'global':
            cross_enc_output = self.crossencoderlayer(ce_pad_input, combined_global_mask)
        elif self.mask == 'cross':
            cross_enc_output = self.crossencoderlayer(ce_pad_input, combined_cross_mask)
        elif self.mask == 'local':
            cross_enc_output = self.crossencoderlayer(ce_pad_input, crombined_local_mask)

        # 对应src-pad-tar-pad
        ce_src_output = cross_enc_output[:, 0:self.max_len,:]
        ce_tar_output = cross_enc_output[:, self.max_len:, :]

        ce_src_output = self.ST_layer2(ce_src_output, src_mask)
        ce_src_output = self.ST_layer2(ce_tar_output, tar_mask)
        

        cross_output = self.compute_distance(ce_src_output, ce_src_output, beta)
 

        final_output = self.l * bi_output + self.r * cross_output
        # end_time = time.time()
        # times = end_time - start_time
        # formatted_times = f"{times:.2f}"
        
        # final_output = bi_output
        # final_output = cross_output

        # print('BE_output:', bi_output)
        # print("CE_output:", cross_output)

        return final_output 
    def make_src_mask(self, src):
        #生成一个[batch_size,1,1,seq_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_local_mask(self, sz, k=4):
        #生成一个[sz,sz]
        mask = torch.eye(sz)
        for i in range(1, k + 1):
            mask += torch.cat((torch.zeros(i, sz), torch.eye(sz)[:-i]), dim=0)
            mask += torch.cat((torch.zeros(sz, i), torch.eye(sz)[:, :-i]), dim=1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_cross_mask_withoutpad(self, seq, src_lengths, tar_lengths):
        # [batch_size, seq_len]
        
        src_mask = self.make_src_mask(seq)
        batch_size, seq_len = seq.size()
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, device = device)

        for i in range(batch_size):
            len1 = src_lengths[i]
            len2 = tar_lengths[i]
            length = len1 + len2
            window_mask = torch.zeros(seq_len, seq_len, device = device)
            window_mask[:len1, len1:length] = 1
            window_mask[len1:length, :len1] = 1
            
            mask[i, 0, :,:] = window_mask

        final_mask = src_mask & (mask == 1)

        return src_mask, final_mask

    def generate_cross_mask_withpad(self, seq):
        # [batch_size, seq_len]
        
        src_mask = self.make_src_mask(seq)
        batch_size, seq_len = seq.size()
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, device = device)

        window_mask = torch.zeros(seq_len, seq_len, device = device)
        window_mask[:self.max_len, self.max_len: 2* self.max_len] = 1
        window_mask[self.max_len: 2* self.max_len, :self.max_len] = 1
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)
        final_mask = src_mask & (window_mask == 1)
        return src_mask, final_mask


    def generate_local_cross_mask(self, seq, src_length, tar_length, k = 3):
        src_mask = self.make_src_mask(seq)
        batch_size, seq_len = seq.size()

        cross_mask = torch.zeros(seq_len, seq_len, device=device)
        cross_mask[:self.max_len, self.max_len: seq_len] = 1
        cross_mask[self.max_len: seq_len, :self.max_len] = 1
        cross_mask = cross_mask.unsqueeze(0).unsqueeze(0)

        local_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device= device)

        for i in range(self.max_len):
            start = max(0, i - k)
            end = min(self.max_len, i + k + 1)
            local_mask[:, 0, i, self.max_len + start:self.max_len + end] = 1
            local_mask[:, 0, self.max_len + i, start:end] = 1
        for i in range(batch_size):
            len1 = src_length[i]
            len2 = tar_length[i]

            local_mask[i, 0, len2 - k - 1:self.max_len, self.max_len + len2 - k - 1: self.max_len + len2] = 1
            local_mask[i, 0, self.max_len + len1 - k - 1:, len1 - k - 1: len1] = 1

        final_mask = src_mask & (local_mask == 1) & (cross_mask == 1)
        cross_mask = src_mask & (cross_mask == 1)
        return src_mask, cross_mask, final_mask
