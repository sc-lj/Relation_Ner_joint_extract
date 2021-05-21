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



class ConditionalLayerNorm(nn.Module):
    """条件layer normal"""
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = nn.Linear(hidden_size , hidden_size, bias=False)
        self.gamma_dense = nn.Linear(hidden_size , hidden_size, bias=False)

    def forward(self, x, cond):
        # cond: [batch_size,1,bert_dim]
        # cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias

class SubLayerNorm(nn.Module):
    """带主语的layer normal"""
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, cond):
        # cond: [batch_size,1,bert_dim]
        # cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias


class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2



        