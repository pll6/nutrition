import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES, HEADS

@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        # 使用 PyTorch 原生的 smooth_l1_loss
        loss = F.smooth_l1_loss(pred, target, beta=self.beta)
        return loss * self.loss_weight

@LOSSES.register_module()
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=5):
        super(MultiTaskUncertaintyLoss, self).__init__()
        # nn.Parameter 告诉 PyTorch：这不是一个普通的张量，
        # 它是需要被优化器更新的参数！
        # 初始值为 0，意味着初始的 sigma^2 = exp(0) = 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets):
        # 1. 计算基础误差 (假设这里 preds 和 targets 都已经做过 Z-score 标准化)
        # reduction='none' 保留每个样本、每个维度的单独误差
        base_losses = F.smooth_l1_loss(preds, targets, reduction='none')
        
        # 将 Batch 维度求均值，保留 5 个维度的误差
        # base_losses 的 shape 变成 [5]
        base_losses = base_losses.mean(dim=0) 
        
        # 2. 不确定性加权
        # precision 等价于公式中的 1 / (2 * sigma^2)
        precision = torch.exp(-self.log_vars)
        
        # 加权 Loss + 正则项
        loss = precision * base_losses + self.log_vars
        
        # 3. 把 5 个维度的 Loss 求和作为最终的单一标量 Loss
        return loss.sum()
