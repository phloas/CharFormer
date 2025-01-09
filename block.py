import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import d_model
from layer import LayerNorm, MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = torch.nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = torch.nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
class SETensorNetworkModule(torch.nn.Module):

    def __init__(self):
        super(SETensorNetworkModule, self).__init__()
        self.d_model = d_model
        self.setup_weights()

    def setup_weights(self):
        channel = self.d_model*2
        reduction = 4
        self.fc_se = torch.nn.Sequential(
                        torch.nn.Linear(channel,  channel // reduction),
                        torch.nn.ReLU(inplace = True),
                        torch.nn.Linear(channel // reduction, channel),
                        torch.nn.Sigmoid()
                )

        self.fc0 = torch.nn.Sequential(
                        torch.nn.Linear(channel, channel),
                        torch.nn.ReLU(inplace = True),
                        torch.nn.Linear(channel, channel),
                        torch.nn.ReLU(inplace = True)
                )

        self.fc1 = torch.nn.Sequential(
                        torch.nn.Linear(channel,  channel),
                        torch.nn.ReLU(inplace = True),
                        torch.nn.Linear(channel, self.d_model // 2), #nn.Linear(channel, self.args.tensor_neurons),
                        torch.nn.ReLU(inplace = True)
                )

    def forward(self, embedding_1, embedding_2):

        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        se_feat_coefs = self.fc_se(combined_representation)
        se_feat = se_feat_coefs * combined_representation + combined_representation
        scores = self.fc1(se_feat)

        return scores
    
class SEAttentionModule(torch.nn.Module):
    def __init__(self, dim):
        super(SEAttentionModule, self).__init__()
        self.dim = dim
        self.setup_weights()

    def setup_weights(self):
        channel = self.dim*1
        reduction = 4
        self.fc = torch.nn.Sequential(
                        torch.nn.Linear(channel,  channel // reduction),
                        torch.nn.ReLU(inplace = True),
                        torch.nn.Linear(channel // reduction, channel),
                        torch.nn.Sigmoid()
                )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class ST_Layer(nn.Module):
    def __init__(self, embedding_size, device):
        super(ST_Layer, self).__init__()
        self.device = device
        # self.bi_lstm = nn.LSTM(input_size=embedding_size,
        #                        hidden_size=hidden_size,
        #                        num_layers=num_layers,
        #                        batch_first=True,
        #                        dropout=dropout_rate,
        #                        bidirectional=True)
        # self-attention weights
        self.w_1 = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.w_2 = nn.Parameter(torch.Tensor(embedding_size, 1))

        nn.init.uniform_(self.w_1, -0.1, 0.1)
        nn.init.uniform_(self.w_2, -0.1, 0.1)

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self, input, mask):
        # [batch_size, seq_len, d_model]
        # [batch_size, 1, 1, seq_len]
        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        v = torch.tanh(torch.matmul(input, self.w_1))
        mask = mask.squeeze(1).squeeze(1)
        # (batch_size, seq_len)
        att = torch.matmul(v, self.w_2).squeeze()

        # add mask
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = input * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        # print(out.size()) 
        return out