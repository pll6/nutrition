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