import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from functools import partial
from utils.registry import register
from .globalpointer import sequence_masking


registry = {}
register = partial(register, registry=registry)

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = y_pred.gt(0).float()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)


class CompLoss(nn.Module):
    def __init__(self,alpha=0.3,beta=0.2):
        super(CompLoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epson = 1e-12
    
    def forward(self,gold,pred,mask):
        weight = torch.zeros_like(gold)
        weight = torch.fill_(weight, self.beta)
        weight[gold > 0] = 1-self.beta
        weight = sequence_masking(weight, mask, 0, 2)
        weight = sequence_masking(weight, mask, 0, 3)
        loss1 = f.binary_cross_entropy(pred,gold,reduction='none')
        loss1 =torch.mean(loss1*weight)

        loss3 = (2*torch.sum(gold*pred)+self.epson)/(torch.sum(pred)+torch.sum(gold)+self.epson)
        loss = 2*loss1 + 10*loss3
        return loss




@register("gpce")
class GlobalCrossEntropy(nn.Module):
    def __init__(self,threshold=0):
        super(GlobalCrossEntropy,self).__init__()
        self.threshold = threshold

    def multilabel_categorical_crossentropy(self,y_true, y_pred,threshold):
        """多标签分类的交叉熵
        说明：
            1. y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类；
            2. 请保证y_pred的值域是全体实数，换言之一般情况下
            y_pred不用加激活函数，尤其是不能加sigmoid或者
            softmax；
            3. 预测阶段则输出y_pred大于0的类；
            4. 详情请看：https://kexue.fm/archives/7359 。
        """
        y_pred = (1 - 2 * y_true) * y_pred # 将标签为1的pred取反
        y_pred_neg = y_pred - y_true * 1e12 # 将标签为1的pred取无限小
        y_pred_pos = y_pred - (1 - y_true) * 1e12 # 将标签为0的pred 取无限小
        thresh = torch.zeros_like(y_pred[..., :1])
        thresh[:] = threshold
        y_pred_neg = torch.cat([y_pred_neg, thresh], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, -thresh], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def forward(self,y_true, y_pred,threshold=0):
        """给GlobalPointer设计的交叉熵
        """
        if y_pred.dim()>3:
            batch_size, rel_num = y_pred.shape[0],y_pred.shape[1]
            bh = batch_size*rel_num
        else:
            bh = y_pred.shape[0]
        y_true = torch.reshape(y_true, (bh, -1))
        y_pred = torch.reshape(y_pred, (bh, -1))
        loss = self.multilabel_categorical_crossentropy(y_true, y_pred,threshold)
        return torch.mean(loss)


@register("ghmc")
class GHMC(nn.Module):
    """GHM Classification Loss.
    Ref:https://github.com/libuyu/mmdetection/blob/master/mmdet/models/losses/ghm_loss.py
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0, alpha=None,**kwargs):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

        self.label_weight = alpha

    def forward(self, pred, target, label_weight=None, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label

        # if pred.dim() != target.dim():
        #     target, label_weight = _expand_binary_labels(
        #     target, label_weight, pred.size(-1))

        # 我的pred输入为[B,C]，target输入为[B]
        target = torch.zeros(target.size(0), 2).to(target.device).scatter_(1, target.view(-1, 1), 1)

        # 暂时不清楚这个label_weight输入形式，默认都为1
        if label_weight is None:
            label_weight = torch.ones([pred.size(0), pred.size(-1)]).to(target.device)

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # sigmoid梯度计算
        g = torch.abs(pred.sigmoid().detach() - target)
        # 有效的label的位置
        valid = label_weight > 0
        # 有效的label的数量
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # 将对应的梯度值划分到对应的bin中， 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # 该bin中存在多少个样本
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment计算num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    # 权重等于总数/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale系数
            weights = weights / n

        loss = f.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight



class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.75, num_classes=2,**kwargs):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self._num_classes = num_classes

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        target = target.view(-1,1)
        # 将target转化为one hot形式
        batch_size = target.shape[0]
        y = torch.zeros((batch_size, self._num_classes), device=target.device)
        y.scatter_(1,target,1)
        g = torch.abs(self._custom_loss_grad(x, y)).detach() # g正比于检测的难易程度，g越大则检测难度越大。

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins),device=target.device)
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, y, beta[bin_idx])



class GHMC_Loss(GHM_Loss):
    def __init__(self, bins=8, alpha=0.9, num_classes=2,**kwargs):
        super(GHMC_Loss, self).__init__(bins, alpha, num_classes)

    def _custom_loss(self, x, target, weight):
        return f.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    def __init__(self, bins=10, alpha=0.75, mu=1e-5, num_classes=2):
        super(GHMR_Loss, self).__init__(bins, alpha, num_classes)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)

@register("focal")
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=0.75, size_average=True,**kwargs):
        """
        gamma: 对困难样本的关注程度
        alpha: 平衡正负样本数量，对于二分类而言，正样本少，属于正样本的alpha应该越大
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([1-alpha,alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target,mask=None):
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        # logpt = f.log_softmax(input, dim=-1)
        # logpt = logpt.gather(1,target)
        logpt = input.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1).long())
            logpt = logpt * Variable(at)
        if mask is not None:
            loss = -1 * (1-pt)**self.gamma * logpt*mask.view(-1)
        else:
            loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

@register("lmcl")
class LMCL_Loss(nn.Module):
    """large margin cosine loss"""
    def __init__(self,scale=2, margin=0.35,**kwargs):
        super(LMCL_Loss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, y_pred, y_true):
        batch_size, class_num = y_pred.shape
        device = y_pred.device
        y_true = y_true.view(-1,1)
        one_hot = torch.zeros(batch_size, class_num,device=device).scatter_(1, y_true, 1)
        y_pred = one_hot * (y_pred-self.margin)+(1-one_hot)*y_pred
        y_pred *= self.scale
        y_true = y_true.view(-1)
        return f.cross_entropy(y_pred, y_true)


def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    """
    Circle Loss
    https://arxiv.org/abs/2002.10857
    """
    def __init__(self, m=0.25, gamma=2,**kwargs):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, logit, target):
        sp, sn = convert_label_to_similarity(logit, target)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == '__main__':
    feat = nn.functional.normalize(torch.rand(256, 2, requires_grad=True))
    print(feat)
    lbl = torch.randint(high=2, size=(256,))
    print(lbl)
    criterion = CircleLoss(m=0.25, gamma=2,device=torch.device("cpu"))
    circle_loss = criterion(feat, lbl)
    print(circle_loss)


