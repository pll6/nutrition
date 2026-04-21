import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES

@LOSSES.register_module()
class WeightedMSELoss(nn.Module):
    """
    第二篇论文 (Feng et al., 2026) 中的加权 MSE Loss
    """
    def __init__(self, train_means, loss_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.loss_weight = loss_weight
        
        means_tensor = torch.tensor(train_means, dtype=torch.float32)
        
        # 计算与均值平方成反比的权重
        weights = 1.0 / (means_tensor ** 2 + 1e-6)
        
        # 将权重归一化到 [0, 1] 之间 (除以最大值)
        weights = weights / weights.max()
        
        self.register_buffer('nutrient_weights', weights)

    def forward(self, pred, target, **kwargs):
        # 1. 计算每个样本、每个营养素的 MSE (不求和)
        mse = torch.nn.functional.mse_loss(pred, target, reduction='none') # [B, 5]
        
        # 2. 乘以固定的归一化权重 (利用广播机制)
        weighted_mse = mse * self.nutrient_weights # [B, 5]
        
        # 3. 对 Batch 和通道维度求平均 (或求和，视你的网络学习率而定)
        return weighted_mse.mean() * self.loss_weight