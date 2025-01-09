# Model Parameters
d_model = 128
n_head = 8

ffn_hidden = 2048
n_layers = 1
drop_prob = 0
# device = torch.device("cuda:0" if torch.cuda.is_avail     able() else "cpu")
device = 'cuda:0'

#exp的参数
# beta = 20
# beta = 10
# ## uniref_500
# beta = 0.80
# ## word
# beta = 1.80
# ## gen50ks_500
# beta = 0.93
# ## gen50ks_100
# beta = 1.13
# ## uniref_name
# beta = 2.17

#sub划分的大小，不同数据集的大小不同
# step = 100
# window_size = 200
# num_samples = 2

scale_factor = 0.03

#seed
seed = 808

# CNNED baseline
mtc_input = False
#CNN的通道数
channel = 8

# LSTM baseline
num_layers = 1

# Training Parameters
nhid = 128
# learning_rate = 0.0001
init_lr = 1e-3
epochs = 10000
batch_size = 50
batch_size_valid = 250
sampling_num = 6

l = 0.9
r = 0.1

K = 100

early_stop_patience = 10
scale_factor = 100

# optimizer parameter setting

## learning_rate 

factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100

clip = 1.0
weight_decay = 5e-4
inf = float('inf')

