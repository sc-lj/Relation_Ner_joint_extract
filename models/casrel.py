from torch import nn
from transformers import *
import random
import torch.nn.functional as F
import torch
import math
from .attention import registry as attention
from .globalpointer import GlobalPointer
from .Loss import FocalLoss,GlobalCrossEntropy


class Casrel(nn.Module):
    def __init__(self, config):
        super(Casrel, self).__init__()
        self.focal_loss = FocalLoss()
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
        sub_heads_loss = self.loss(data['sub_heads'], pred_sub_heads, data['mask'],True)
        sub_tails_loss = self.loss(data['sub_tails'], pred_sub_tails, data['mask'],True)
        obj_heads_loss = self.loss(data['obj_heads'], pred_obj_heads, data['mask'])
        obj_tails_loss = self.loss(data['obj_tails'], pred_obj_tails, data['mask'])
        total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)
        # return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
        return sub_heads_loss,sub_tails_loss,obj_heads_loss,obj_tails_loss

    # define the loss function
    def loss(self,gold, pred, mask,use_focal=False):
        pred = pred.squeeze(-1)
        los = F.binary_cross_entropy(pred, gold, reduction='none')
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los * mask) / torch.sum(mask)
        if self.config.use_focal and use_focal:
            los += self.focal_loss(pred,gold,None)
        return los


class GlobalPointerRel(nn.Module):
    def __init__(self, config):
        super(GlobalPointerRel, self).__init__()
        self.config = config
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_path)
        self.bert_dim = self.bert_encoder.config.hidden_size
        # self.obj_heads_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        # self.obj_tails_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.sub_global = GlobalPointer(1,self.config.head_size,self.bert_dim)
        self.obj_global = GlobalPointer(self.config.rel_num,self.config.head_size, self.bert_dim)


    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text,mask = None):
        # [batch_size, 1, bert_dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2
        # [batch_size, seq_len, bert_dim]
        encoded_text = encoded_text + sub
        # [batch_size, rel_num, seq_len, seq_len]
        pred_objs = self.obj_global(encoded_text,mask)
        return pred_objs

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text,mask):
        # [batch_size,seq_len,seq_len]
        pred_subs = self.sub_global(encoded_text,mask).squeeze(1)
        return pred_subs

    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size,seq_len,seq_len]
        pred_subs = self.get_subs(encoded_text, mask)
        # [batch_size, 1, seq_len]
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # [batch_size, rel_num, seq_len, seq_len]
        pred_objs = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text, mask)
        sub_loss = self.pointer_loss(data['pointer_sub'], pred_subs)
        # pred_subs = torch.sigmoid(pred_subs)
        # sub_loss = self.pointer_sub_loss(data['pointer_sub'], pred_subs,True)
        obj_loss = self.pointer_loss(data['pointer_obj'], pred_objs)
        total_loss = 1.2*sub_loss+1.*obj_loss
        # return pred_subs, pred_objs
        return sub_loss,obj_loss

    def pointer_loss(self,gold,pred,threshold=0):
        loss_func = GlobalCrossEntropy()
        los = loss_func(gold,pred,threshold)
        return los
    def pointer_sub_loss(self,gold,pred,use_focal=False):
        los = F.binary_cross_entropy(pred, gold)
        if self.config.use_focal and use_focal:
            los += self.focal_loss(pred,gold)
        return los