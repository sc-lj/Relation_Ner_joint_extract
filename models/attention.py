# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from utils.registry import register

registry = {}
register = partial(register, registry=registry)


@register('identity')
class Alignment(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()

    def forward(self, head,tail):
        # 基于head 对tail的attention，并预测tail
        temperature = nn.Parameter(torch.tensor(1 / math.sqrt(tail.shape[2])))
        # [batch_size, seq_len, seq_len]
        weight = torch.matmul(head,tail.transpose(2,1))/temperature  # head -> encoded_tex attention
        weight = F.softmax(weight,dim=2)
        # [batch_size, seq_len, 1]
        pred_tails = torch.matmul(weight,tail)
        return pred_tails


@register('linear')
class MappedAlignment(nn.Module):
    def __init__(self, args, input_size,output_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(input_size, output_size, activations=True),
        )

    def forward(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        temperature = nn.Parameter(torch.tensor(1 / math.sqrt(b.shape[2])))
        attn = torch.matmul(a, b.transpose(1, 2)) * temperature
        # mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        # attn.masked_fill_(~mask.bool(), -1e7)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b

@register('None')
class NoneAlignment(nn.Module):
    def __init__(self, args, input_size):
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