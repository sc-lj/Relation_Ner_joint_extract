from torch import nn
from transformers import *
import random
import torch.nn.functional as F
import torch
import math
from .attention import registry as attention


class Casrel(nn.Module):
    def __init__(self, config):
        super(Casrel, self).__init__()
        self.config = config
        self.bert_dim = 768
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_path)
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.obj_tails_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.sub_attention = attention[self.config.attention](config,1,64)
        self.obj_attention = attention[self.config.attention](config,self.config.rel_num,self.config.rel_num)


    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # [batch_size, 1, bert_dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2
        # [batch_size, seq_len, bert_dim]
        encoded_text = encoded_text + sub
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        # [batch_size, seq_len, rel_num]
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        # add attention
        # if self.config.attention == "head2tail":
        #     pred_obj_heads,pred_obj_tails = self.obj_attention(pred_obj_heads,pred_obj_tails)
        # else:
        #     pred_obj_tails = self.obj_attention(pred_obj_heads,pred_obj_tails)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text)
        # add attention
        if self.config.attention == "head2tail":
            pred_sub_heads,pred_sub_tails = self.sub_attention(pred_sub_heads,pred_sub_tails)
        else:
            pred_sub_tails = self.sub_attention(pred_sub_heads,pred_sub_tails)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        # if random.random()>self.config.teacher_pro: # teacher probability
        #     # [batch_size, 1, seq_len]
        #     sub_head = pred_sub_heads.permute(0,2,1)
        #     # [batch_size, 1, seq_len]
        #     sub_tail = pred_sub_tails.permute(0,2,1)
        #     if random.random()<0.5:
        #         # [batch_size, 1, 1]
        #         sub_heads, sub_tails = sub_head.argmax(-1,keepdim=True), sub_tail.argmax(-1,keepdim=True)
        #         # [batch_size, 1, seq_len]
        #         sub_head_mapping = torch.zeros_like(sub_head,dtype=sub_head.dtype)
        #         # [batch_size, 1, seq_len]
        #         sub_tail_mapping = torch.zeros_like(sub_tail,dtype=sub_tail.dtype)
        #         sub_head_mapping.scatter_(-1,sub_heads,1)
        #         sub_tail_mapping.scatter_(-1,sub_tails,1)
        #         fuse = (sub_heads>sub_tails).view(-1)
        #         sub_head_mapping[fuse] = 0
        #         sub_tail_mapping[fuse] = 0
        #     else:
        #         sub_head_mapping = sub_head
        #         sub_tail_mapping = sub_tail
        # else:
        # [batch_size, 1, seq_len]
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
