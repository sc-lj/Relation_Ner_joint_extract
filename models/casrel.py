from torch import nn
from transformers import *
import random
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd,autocast
import torch
import math
from .attention import registry as attention
from .attention import ConditionalLayerNorm,Linear,MultiNonLinearClassifier
from .globalpointer import GlobalPointer,JointGlobalPointer,sequence_masking
from .Loss import FocalLoss,GlobalCrossEntropy,CompLoss


class FullFusion(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size * 2, input_size, activations=True)
        self.fusion2 = Linear(input_size * 2, input_size, activations=True)
        self.fusion = Linear(input_size * 2, input_size, activations=True)

    def forward(self, encoded_text,sub):
        x1 = self.fusion1(torch.cat([encoded_text, encoded_text - sub], dim=-1))
        x2 = self.fusion2(torch.cat([encoded_text, encoded_text * sub], dim=-1))
        encoded_text = torch.cat([x1, x2], dim=-1)
        encoded_text = F.dropout(encoded_text, self.dropout, self.training)
        return self.fusion(encoded_text)


class BaseModel(nn.Module):
    def __init__(self,config):
        super(BaseModel, self).__init__()
        self.focal_loss = FocalLoss()
        self.config = config
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_path)
        self.bert_dim = self.bert_encoder.config.hidden_size
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.obj_tails_linear = nn.Linear(self.bert_dim, self.config.rel_num)
        self.head_inter_tail = config.head_inter_tail
        self.rel_matrix =nn.Parameter(torch.Tensor(self.config.rel_num,self.config.head_size))
        self.sub2encode = FullFusion(config,self.bert_dim)
        if self.config.identiy:
            self.layernormal = ConditionalLayerNorm(self.bert_dim)
        else:
            self.layernormal = ConditionalLayerNorm(self.bert_dim*4)
        nn.init.kaiming_normal_(self.rel_matrix)


    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text
    

    def get_subs(self, encoded_text,mask=None):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text+pred_sub_heads)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    # define the loss function
    def sub_loss(self,gold, pred, mask,weight=1,use_focal=False):
        pred = pred.squeeze(-1)
        weights = torch.zeros_like(gold)
        weights = torch.fill_(weights, 1)
        weights[gold > 0] = weight
        los = F.binary_cross_entropy(pred, gold, reduction='none')
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los *weights* mask) / torch.sum(mask)
        if self.config.use_focal and use_focal:
            los += self.focal_loss(pred,gold,None)
        return los
    
    
    def get_range_head_tail(self,head,tail):
        """给定头尾索引的矩阵，给出头尾索引之间有填充后的矩阵"""
        # [batch,seq_len]
        head = head.squeeze(1)
        tail = tail.squeeze(1)
        # [batch,1]
        values = (head + tail).mean(-1,keepdim=True)
        head_index = torch.nonzero(head)[:,1] #非零元素的index
        tail_index = torch.nonzero(tail)[:,1]
        mask = torch.zeros_like(head)
        seq_len = head.shape[-1]
        mask[:,:]= torch.arange(seq_len,device=head.device)
        start_range = (mask<head_index.reshape(-1,1)).int() #将开始到头部index之间的值填充为1
        end_range = (mask<=tail_index.reshape(-1,1)).int() #将开始到尾部index之间的值填充为1
        new_mask = start_range+end_range
        new_mask = (new_mask==1).float() #将头部到尾部index之间的值填充为1
        new_mask *= values
        return new_mask.unsqueeze(1).type_as(head)


    def sub2obj(self,encoded_text,sub_head_mapping,sub_tail_mapping):
        assert not self.head_inter_tail or not self.config.identiy, "`head_inter_tail`和`identiy`两个参数不能同时为True"
        if self.head_inter_tail:
            # [batch_size, 1, seq_len]
            mix_matrix = self.get_range_head_tail(sub_head_mapping,sub_tail_mapping)
            # [batch_size, 1, 1]
            mask_matrix = mix_matrix.sum(-1,keepdim=True)
            # [batch_size, 1, bert_dim]
            sub = torch.matmul(mix_matrix, encoded_text)/mask_matrix
        else:
            # [batch_size, 1, bert_dim]
            sub_head = torch.matmul(sub_head_mapping, encoded_text)
            # [batch_size, 1, bert_dim]
            sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
            # [batch_size, 1, bert_dim]
            sub = (sub_head + sub_tail) / 2

        if self.config.add_layernormal:
            if self.config.identiy:
                sub = torch.cat([sub,sub_head-sub_tail,sub_head,sub_tail])
            # 通过layernormal的形式将让subject与encode进行交互
            encoded_text = self.layernormal(encoded_text,sub)
        elif self.config.fusion:
            encoded_text = self.sub2encode(encoded_text,sub)
        else:
            # [batch_size, seq_len, bert_dim]
            encoded_text = encoded_text + sub
        return encoded_text
    
    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        encoded_text = self.sub2obj(encoded_text,sub_head_mapping,sub_tail_mapping)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        # [batch_size, seq_len, rel_num]
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails


class Casrel(BaseModel):
    def __init__(self,config):
        super(Casrel,self).__init__(config)


    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        if random.random()<self.config.teacher_pro: # teacher probability
            # [batch_size, 1, seq_len]
            sub_head_mapping = pred_sub_heads.permute(0,2,1)*data['sub_head'].unsqueeze(1)
            # [batch_size, 1, seq_len]
            sub_tail_mapping = pred_sub_tails.permute(0,2,1) *data['sub_tail'].unsqueeze(1)
        else:
            # [batch_size, 1, seq_len]
            sub_head_mapping = data['sub_head'].unsqueeze(1)
            # [batch_size, 1, seq_len]
            sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)
        sub_heads_loss = self.sub_loss(data['sub_heads'], pred_sub_heads, data['mask'],2,True)
        sub_tails_loss = self.sub_loss(data['sub_tails'], pred_sub_tails, data['mask'],2,True)
        obj_heads_loss = self.sub_loss(data['obj_heads'], pred_obj_heads, data['mask'],2)
        obj_tails_loss = self.sub_loss(data['obj_tails'], pred_obj_tails, data['mask'],2)
        sub_loss = sub_heads_loss + sub_tails_loss
        obj_loss = obj_heads_loss + obj_tails_loss
        return sub_loss,obj_loss


class GlobalPointerRel(BaseModel):
    def __init__(self, config):
        super(GlobalPointerRel, self).__init__(config)
        self.config = config
        self.sub_global = GlobalPointer(1,self.config.head_size,self.bert_dim)
        self.obj_global = GlobalPointer(self.config.rel_num,self.config.head_size, self.bert_dim)

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text,mask = None):
        encoded_text = self.sub2obj(encoded_text,sub_head_mapping,sub_tail_mapping)
        # [batch_size, rel_num, seq_len, seq_len]
        pred_objs = self.obj_global(encoded_text,mask)
        return pred_objs

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
        # [batch_size, seq_len]
        sub_head_mapping = data['sub_head']
        # [batch_size, seq_len]
        sub_tail_mapping = data['sub_tail']

        if self.config.teacher_pro <= random.random():
            # 将主语的部分信息带入到下游，谓语预测中
            special_sub_logit = self.get_special_sub_logit(sub_head_mapping,sub_tail_mapping,pred_subs)
            sub_head_mapping = sub_head_mapping*special_sub_logit
            sub_tail_mapping = sub_tail_mapping*special_sub_logit

        # [batch_size, 1, seq_len]
        sub_head_mapping = sub_head_mapping.unsqueeze(1)
        sub_tail_mapping = sub_tail_mapping.unsqueeze(1)

        # [batch_size, rel_num, seq_len, seq_len]
        pred_objs = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text, mask)
        sub_loss = self.pointer_loss(data['pointer_sub'], pred_subs)
        # pred_subs = torch.sigmoid(pred_subs)
        # sub_loss = self.pointer_sub_loss(data['pointer_sub'], pred_subs,True)
        obj_loss = self.pointer_loss(data['pointer_obj'], pred_objs)
        # total_loss = 1.2*sub_loss+1.*obj_loss
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

    def get_special_sub_logit(self,head,tail,logit):
        relu = nn.ReLU()
        head_index = torch.nonzero(head)[:,1]
        tail_index = torch.nonzero(tail)[:,1]
        head_tail = torch.vstack((head_index,tail_index)).T

        # [batch,1]
        special_sub_logit = self.torch_gather_nd(logit,head_tail).unsqueeze(1)
        special_sub_logit = relu(special_sub_logit) # 采用relu激活函数
        return special_sub_logit
    
    def torch_gather_nd(self,logit,head_tail):
        batch = logit.shape[0]
        idx_chunked = head_tail.chunk(2,1) # 对张量分块,
        masked = logit[torch.arange(batch).view(batch,1),idx_chunked[0].squeeze(),idx_chunked[1].squeeze()]
        diag = torch.diag(masked)
        return diag
    
    def torch_gather_nd_(self,x, head_tail):
        "给定所有索引，获取张量中的值"
        batch = x.shape[0]
        index = torch.hstack([torch.arange(batch,device=x.device,dtype=torch.float).reshape(-1,1),head_tail])
        x = x.contiguous()
        with autocast(enabled=False):
            stride = torch.tensor(x.stride(),device=x.device,dtype=torch.float)
            inds = torch.matmul(index,stride)
            # inds = inds.long() 
            x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
        return x_gather


class CasGlobal(BaseModel):
    def __init__(self, config):
        super(CasGlobal, self).__init__(config)
        self.pointer_loss = GlobalCrossEntropy()

    def get_subs(self, encoded_text,mask):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text+pred_sub_heads)
        pred_sub_heads = sequence_masking(pred_sub_heads,mask)
        pred_sub_tails = sequence_masking(pred_sub_tails,mask)
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text,mask):
        encoded_text = self.sub2obj(encoded_text,sub_head_mapping,sub_tail_mapping)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        # [batch_size, seq_len, rel_num]
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        # [batch_size, seq_len, 1]
        mask = mask.unsqueeze(-1)
        # [batch_size, seq_len, rel_num]
        mask = mask.repeat(1,1,self.config.rel_num)
        pred_obj_heads = sequence_masking(pred_obj_heads,mask)
        pred_obj_tails = sequence_masking(pred_obj_tails,mask)
        return pred_obj_heads, pred_obj_tails
        
    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size,seq_len,1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text,mask)
        # [batch_size, seq_len]
        sub_head_mapping = data['sub_head']
        # [batch_size, seq_len]
        sub_tail_mapping = data['sub_tail']

        if self.config.teacher_pro <= random.random():
            # 将主语的部分信息带入到下游，谓语预测中
            # [batch_size, 1, seq_len]
            sub_head_mapping = pred_sub_heads.permute(0,2,1)*data['sub_head'].unsqueeze(1)
            # [batch_size, 1, seq_len]
            sub_tail_mapping = pred_sub_tails.permute(0,2,1) *data['sub_tail'].unsqueeze(1)
        else:
            # [batch_size, 1, seq_len]
            sub_head_mapping = sub_head_mapping.unsqueeze(1)
            sub_tail_mapping = sub_tail_mapping.unsqueeze(1)

        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text, mask)
        sub_heads_loss = self.pointer_loss(data['sub_heads'], pred_sub_heads)
        sub_tails_loss = self.pointer_loss(data['sub_tails'], pred_sub_tails)
        sub_loss = sub_heads_loss + sub_tails_loss
        obj_heads_loss = self.pointer_loss(data['obj_heads'], pred_obj_heads)
        obj_tails_loss = self.pointer_loss(data['obj_tails'], pred_obj_tails)
        obj_loss = obj_heads_loss + obj_tails_loss
        return sub_loss,obj_loss


class CasGlobalPointer(GlobalPointerRel):
    def __init__(self, config):
        super(CasGlobalPointer, self).__init__(config)
        self.focal_loss = FocalLoss()

    def get_encoded_text(self,token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_output = self.bert_encoder(token_ids, attention_mask=mask,return_dict=True,output_hidden_states=True)
        encoded_text = encoded_output['last_hidden_state']
        subject_encoded_text = encoded_output["hidden_states"][6]
        # encoded_text = (encoded_text+subject_encoded_text)/2
        return encoded_text,subject_encoded_text


    def get_subs(self, encoded_text,mask=None):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text+pred_sub_heads)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

        
    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text,subject_encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size,seq_len,1]
        pred_sub_heads, pred_sub_tails = self.get_subs(subject_encoded_text)
        # [batch_size, seq_len]
        sub_head_mapping = data['sub_head']
        # [batch_size, seq_len]
        sub_tail_mapping = data['sub_tail']

        if self.config.teacher_pro <= random.random():
            # 将主语的部分信息带入到下游，谓语预测中
            # [batch_size, 1, seq_len]
            sub_head_mapping = pred_sub_heads.permute(0,2,1)*data['sub_head'].unsqueeze(1)
            # [batch_size, 1, seq_len]
            sub_tail_mapping = pred_sub_tails.permute(0,2,1) *data['sub_tail'].unsqueeze(1)
        else:
            # [batch_size, 1, seq_len]
            sub_head_mapping = sub_head_mapping.unsqueeze(1)
            sub_tail_mapping = sub_tail_mapping.unsqueeze(1)

        # [batch_size, rel_num, seq_len, seq_len]
        pred_objs = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text, mask)
        sub_heads_loss = self.sub_loss(data['sub_heads'], pred_sub_heads, data['mask'],True)
        sub_tails_loss = self.sub_loss(data['sub_tails'], pred_sub_tails, data['mask'],True)
        sub_loss = sub_heads_loss + sub_tails_loss
        obj_loss = self.pointer_loss(data['pointer_obj'], pred_objs)
        # return pred_subs, pred_objs
        return sub_loss,obj_loss


class CasNewSub(BaseModel):
    def __init__(self, config):
        super(CasNewSub, self).__init__(config)
        self.focal_loss = FocalLoss()
        self.span_embedding = MultiNonLinearClassifier(self.bert_dim * 2, 1, config.dropout)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')


    def get_encoded_text(self,token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_output = self.bert_encoder(token_ids, attention_mask=mask,return_dict=True,output_hidden_states=True)
        encoded_text = encoded_output['last_hidden_state']
        subject_encoded_text = encoded_output["hidden_states"][6]
        # encoded_text = (encoded_text+subject_encoded_text)/2
        return encoded_text,subject_encoded_text


    def get_subs(self, encoded_text):
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        seq_len = encoded_text.shape[1]
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text+pred_sub_heads)
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = encoded_text.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = encoded_text.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        pred_sub_span = self.span_embedding(span_matrix).squeeze(-1)
        # pred_sub_heads = torch.sigmoid(pred_sub_heads)
        # pred_sub_tails = torch.sigmoid(pred_sub_tails)
        # pred_sub_span = torch.sigmoid(pred_sub_span)
        return pred_sub_heads, pred_sub_tails, pred_sub_span

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        start_logits = start_logits.squeeze()
        end_logits = end_logits.squeeze()
        batch_size, seq_len = start_logits.size()
        start_labels = start_labels.bool()
        end_labels = end_labels.bool()
        match_labels = match_labels.bool()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if self.config.teacher_pro <= random.random():
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                    & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                    & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss
        
    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text,subject_encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size,seq_len,1]
        pred_sub_heads, pred_sub_tails, pred_sub_span = self.get_subs(subject_encoded_text)
        # [batch_size, seq_len]
        sub_head_mapping = data['sub_head']
        # [batch_size, seq_len]
        sub_tail_mapping = data['sub_tail']

        if self.config.teacher_pro <= random.random():
            # 将主语的部分信息带入到下游，谓语预测中
            # [batch_size, 1, seq_len]
            sub_head_mapping = pred_sub_heads.permute(0,2,1)*data['sub_head'].unsqueeze(1)
            # [batch_size, 1, seq_len]
            sub_tail_mapping = pred_sub_tails.permute(0,2,1) *data['sub_tail'].unsqueeze(1)
        else:
            # [batch_size, 1, seq_len]
            sub_head_mapping = sub_head_mapping.unsqueeze(1)
            sub_tail_mapping = sub_tail_mapping.unsqueeze(1)

        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping, encoded_text)
        sub_heads_loss, sub_tails_loss, match_loss = self.compute_loss(pred_sub_heads, pred_sub_tails, pred_sub_span, data['sub_heads'], data['sub_tails'], data['pointer_sub'], data['mask'], data['mask'])
        sub_loss = sub_heads_loss + sub_tails_loss + match_loss

        obj_heads_loss = self.sub_loss(data['obj_heads'], pred_obj_heads, data['mask'])
        obj_tails_loss = self.sub_loss(data['obj_tails'], pred_obj_tails, data['mask'])
        obj_loss = obj_heads_loss + obj_tails_loss
        # return pred_subs, pred_objs
        return sub_loss,obj_loss


class JointSubObj(nn.Module):
    def __init__(self,config):
        super(JointSubObj, self).__init__()
        self.focal_loss = FocalLoss()
        self.config = config
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_path)
        self.bert_dim = self.bert_encoder.config.hidden_size
        self.tails_linear = nn.Linear(self.bert_dim, self.bert_dim)
        self.sub_obj = JointGlobalPointer(self.config.rel_num,self.config.head_size, self.bert_dim)
        self.m1 = nn.LayerNorm(self.bert_dim)

    def get_encoded_text(self,token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_output = self.bert_encoder(token_ids, attention_mask=mask,return_dict=True,output_hidden_states=True)
        encoded_text = encoded_output['last_hidden_state']
        subject_encoded_text = encoded_output["hidden_states"][12]
        # encoded_text = (encoded_text+subject_encoded_text)/2
        return encoded_text,subject_encoded_text
    
    def get_logit(self,encoded_text,subject_encoded_text=None,mask=None):
        # [batch_size, seq_len, bert_dim(768)]
        objects = self.tails_linear(encoded_text)
        objects = F.relu(objects)
        logits = self.sub_obj(subject_encoded_text,objects,mask=mask)
        m2 = nn.LayerNorm(logits.size()[2:],elementwise_affine=False)
        logits = m2(logits)
        return logits


    def forward(self,data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text,subject_encoded_text = self.get_encoded_text(token_ids, mask)
        logits = self.get_logit(encoded_text,encoded_text,mask)
        # loss = self.pointer_loss(data['joint'], logits)
        logits =  torch.sigmoid(logits)
        # loss = self.sub_loss(data['joint'], logits,mask)
        loss = self.combloss(data['joint'], logits,mask)
        return loss

    def pointer_loss(self,gold,pred,threshold=0):
        loss_func = GlobalCrossEntropy()
        los = loss_func(gold,pred,threshold)
        return los

    # define the loss function
    def sub_loss(self,gold, pred, mask):
        los = F.binary_cross_entropy(pred, gold, reduction='none')
    
        weight = torch.zeros_like(gold)
        weight = torch.fill_(weight, 0.2)
        weight[gold > 0] = 0.8
        weight = sequence_masking(weight, mask, 0, 2)
        weight = sequence_masking(weight, mask, 0, 3)

        los = sequence_masking(los, mask, 0, 2)
        los = sequence_masking(los, mask, 0, 3)
        # los = torch.mean(los*weight) 
        los = torch.sum(los*weight) 
        return los
    
    def combloss(self,gold,pred,mask):
        loss_func = CompLoss()
        los = loss_func(gold,pred,mask)
        return los
    

