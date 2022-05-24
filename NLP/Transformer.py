import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

src_len = 124 # enc_input max sequence length
tgt_len = 124 # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = 64 # dimension of K(=Q), 
d_v = 128  # dimension of V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention