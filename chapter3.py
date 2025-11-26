

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length)))

    def forward(self, x):
        b, num_tokens, d_in = x.shape #New batch dimension b
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores = attn_scores.masked_fill(self.mask == 0, -float("inf"))
        attn_weights = nn.functional.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = 4
d_in = 256
d_out = 256
batch_size = 8
num_tokens = 4
x = torch.randn(batch_size, num_tokens, d_in)
causal_attn = CausalAttention(d_in, d_out, context_length, 0.1)
context_vec = causal_attn(x)
print(context_vec)
print(context_vec.shape)

        
