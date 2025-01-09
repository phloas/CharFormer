import torch
import torch.nn as nn
from conf import *

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """
    # Using nn.Embedding

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model 字符向量的维度
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx= 0)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.weight)
    #     if torch.isnan(self.weight).any():
    #         raise ValueError("Embedding layer initialized with NaN values")
        
    def forward(self, x):
        tok_emb = super(TokenEmbedding, self).forward(x)
        if torch.isnan(tok_emb).any():
            print("NaN encountered in token embeddings")
            raise ValueError("NaN encountered in TokenEmbedding output.")
        return tok_emb
        

class OneHotEmbedding:

    def __init__(self, vocab_size):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model 字符向量的维度
        """
        self.vocab_size = vocab_size


    def index_to_one_hot(self, x):
        x = torch.cat((torch.eye(self.vocab_size, device=device), torch.zeros((1, self.vocab_size), device=device)), dim=0)[x]
        return x

    def forward(self, texts):
        one_hot_dic = [self.index_to_one_hot(seq) for seq in texts]
        return torch.stack(one_hot_dic)

class AbsolutePositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    Absolute positional encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(AbsolutePositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)

        self.encoding = torch.zeros(max_len , d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)

        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
       
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(RelativePositionalEmbedding, self).__init__()
        # 【-max_len + 1, max_len - 1】
        self.relative_positions = nn.Embedding(2 * max_len - 1, d_model)
        self.max_len = max_len

    def forward(self, x):
        # x [batch_size, seq_len]
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # 计算每一个token对于其他位置token的位置，转置相减
        relative_positions = positions - positions.transpose(0, 1)
        # 将每一个值增加self.max_len -1 确保范围从[-max_len + 1, max_len - 1] 
        relative_positions += self.max_len - 1
        # 计算每一个位置的向量表示
        relative_embeddings = self.relative_positions(relative_positions)
        return relative_embeddings

class TransformerEmbedding(nn.Module):
    """
    character embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = AbsolutePositionalEncoding(d_model, max_len, device)
        # self.pos_emb = RelativePositionalEmbedding(max_len, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        
        return self.drop_out(tok_emb + pos_emb)



