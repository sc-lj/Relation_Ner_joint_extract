import torch.nn as nn
import torch
from transformers import BertModel


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断,
    详情请看：https://kexue.fm/archives/8373
    """
    def __init__(self, heads, head_size, bert_dim,rel_weight=None, RoPE=True, **kwargs):
        # heads 一般设置为关系数量或者实体类别数量
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.bert_dim = bert_dim
        self.head_size = head_size
        self.RoPE = RoPE
        self.rel_weight = rel_weight
        self.dense = nn.Linear(self.bert_dim,self.head_size * self.heads*2)
        self.position = SinusoidalPositionEmbedding(self.head_size, 'zero')

    def forward(self,inputs,mask=None):
        # [batch_size,seq_len,head_size * heads*2]
        inputs = self.dense(inputs)
        inputs = torch.split(inputs,self.head_size*2,dim = -1)
        # [batch_size,seq_len,heads,head_size*2]
        inputs = torch.stack(inputs, axis=-2)
        # [batch_size,seq_len,heads,head_size]
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # [1,seq_len,head_size]
        pos = self.position(inputs)
        # [1,seq_len,1,head_size],进行元素级别的复制，[1,2]->[1,1,2,2]
        cos_pos = torch.reshape(pos[..., 1::2].unsqueeze(-1).expand(-1,-1,-1,2),pos.shape).unsqueeze(2)
        sin_pos = torch.reshape(pos[..., ::2].unsqueeze(-1).expand(-1,-1,-1,2),pos.shape).unsqueeze(2)
        # [batch_size,seq_len,heads,head_size//2,2]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
        # [batch_size,seq_len,heads,head_size]
        qw2 = torch.reshape(qw2, qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        # [batch_size,seq_len,heads,head_size//2,2]
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
        # [batch_size,seq_len,heads,head_size]
        kw2 = torch.reshape(kw2, kw.shape)
        # kw = kw * cos_pos + kw2 * sin_pos
        # [batch_size,heads,seq_len,head_size]
        qw = qw.permute(0,2,1,3)
        # 计算内积
        # [batch_size,heads,seq_len,seq_len]
        logits = torch.matmul(qw,kw.permute(0,2,3,1))
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角,不排除对角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        # scale返回
        # [batch_size,heads,seq_len,seq_len]
        return logits / self.head_size**0.5



class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, output_dim, merge_mode='add', **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.float,device=inputs.device).reshape(1,-1)
        
        indices = torch.arange(0, self.output_dim // 2, dtype=torch.float,device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        # [1,seq_len,output_dim//2]
        embeddings = torch.matmul(position_ids.unsqueeze(-1),indices.unsqueeze(0))
        # [1,seq_len,output_dim//2,2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # [1,seq_len,output_dim]
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            embeddings = embeddings.repeat([batch_size, 1, 1])
            return torch.cat([inputs, embeddings],dim=-1)


def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    if value == '-inf':
        value = -1e12
    elif value == 'inf':
        value = 1e12
    if axis is None:
        axis = 1
    assert axis > 0, 'axis must be greater than 0'
    for _ in range(axis - 1):
        mask = mask.unsqueeze(1)
    for _ in range(x.dim()- mask.dim()):
        mask = mask.unsqueeze(-1)
    return x * mask + value * (1 - mask)


class JointGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断,
    详情请看：https://kexue.fm/archives/8373
    """
    def __init__(self, heads, head_size, bert_dim, **kwargs):
        # heads 一般设置为关系数量或者实体类别数量
        super(JointGlobalPointer, self).__init__()
        self.heads = heads
        self.bert_dim = bert_dim
        self.head_size = head_size
        self.linear_head = nn.Linear(self.bert_dim,self.head_size * self.heads)
        self.linear_tail = nn.Linear(self.bert_dim,self.head_size * self.heads)
        self.position = SinusoidalPositionEmbedding(self.head_size, 'zero')

    def forward(self,head,tail,mask=None):
        batch_size = head.size(0)
        # [batch_size,seq_len,heads,head_size]
        head = self.linear_head(head).view(batch_size,-1,self.heads,self.head_size)
        tail = self.linear_tail(tail).view(batch_size,-1,self.heads,self.head_size)
        # [1,seq_len,head_size]
        pos = self.position(head)
        # [1,seq_len,1,head_size],进行元素级别的复制，[1,2]->[1,1,2,2]
        cos_pos = torch.reshape(pos[..., 1::2].unsqueeze(-1).expand(-1,-1,-1,2),pos.shape).unsqueeze(2)
        sin_pos = torch.reshape(pos[..., ::2].unsqueeze(-1).expand(-1,-1,-1,2),pos.shape).unsqueeze(2)
        # [batch_size,seq_len,heads,head_size//2,2]
        head2 = torch.stack([-head[..., 1::2], head[..., ::2]], 4)
        # [batch_size,seq_len,heads,head_size]
        head2 = torch.reshape(head2, head.shape)
        head = head * cos_pos + head2 * sin_pos
        # [batch_size,seq_len,heads,head_size//2,2]
        tail2 = torch.stack([-tail[..., 1::2], tail[..., ::2]], 4)
        # [batch_size,seq_len,heads,head_size]
        tail2 = torch.reshape(tail2, tail.shape)
        tail = tail * cos_pos + tail2 * sin_pos
        # [batch_size,heads,seq_len,head_size]
        head = head.permute(0,2,1,3)

        # 计算内积
        # [batch_size,heads,seq_len,seq_len]
        logits = torch.matmul(head,tail.permute(0,2,3,1))
        # 排除padding
        # logits = sequence_masking(logits, mask, '-inf', 2)
        # logits = sequence_masking(logits, mask, '-inf', 3)
        # scale返回
        # [batch_size,heads,seq_len,seq_len]
        return logits / self.head_size**0.5

