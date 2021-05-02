# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.registry import register

registry = {}
register = partial(register, registry=registry)


@register('head_att_tail')
class Attention(nn.Module):
    def __init__(self, *args,**kwargs):
        super().__init__()

    def forward(self, head,tail):
        # 基于head 对tail的attention，并预测tail
        temperature = nn.Parameter(torch.tensor(1 / math.sqrt(tail.shape[2])))
        # [batch_size, seq_len, seq_len]
        weight = torch.matmul(tail,head.transpose(2,1))*temperature  # head -> encoded_tex attention
        weight = F.softmax(weight,dim=2)
        # [batch_size, seq_len, tail.shape[2]]
        pred_tails = torch.matmul(weight,tail)
        return pred_tails


@register('head2tail')
class MappedAttention(nn.Module):
    def __init__(self, config, input_size,output_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Dropout(config.dropout),
            Linear(input_size, output_size, activations=True),
        )
        self.proj2 = Linear(output_size, input_size)

    def forward(self, head, tail):
        head = self.projection(head)
        tail = self.projection(tail)
        temperature = nn.Parameter(torch.tensor(1 / math.sqrt(tail.shape[2])))
        attn = torch.matmul(head, tail.transpose(1, 2)) * temperature
        # mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        # attn.masked_fill_(~mask.bool(), -1e7)
        attn_head = F.softmax(attn, dim=1)
        attn_tail = F.softmax(attn, dim=2)
        pred_tail = torch.matmul(attn_head.transpose(1, 2), head)
        pred_head = torch.matmul(attn_tail, tail)
        pred_head, pred_tail = self.proj2(pred_head),self.proj2(pred_tail)
        return pred_head, pred_tail


@register('None')
class NoneAttention(nn.Module):
    def __init__(self, *args,**kwargs):
        super().__init__()

    def forward(self, a, b):
        return b

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

