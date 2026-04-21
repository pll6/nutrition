import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES

@LOSSES.register_module()
class LogL1Loss(nn.Module):
    """
    Log-L1 Loss 专用于物理量回归。
    通过在对数域计算 L1 误差，模型会天然地去优化 PMAE（相对百分比误差），
    而不是被极大值的绝对误差牵着鼻子走。
    """
    def __init__(self, loss_weight=1.0):
        super(LogL1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        # 加上 1.0 防止 log(0)，同时对极小值起到平滑作用
        log_pred = torch.log(pred + 1.0)
        log_target = torch.log(target + 1.0)
        
        # 使用 L1 Loss 计算对数域的差值，自带鲁棒性，不怕异常大值
        loss = torch.nn.functional.l1_loss(log_pred, log_target)
        
        return loss * self.loss_weight