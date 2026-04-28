import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..builder import TEXT_ENCODERS

@TEXT_ENCODERS.register_module()
class VisualProjectionBranch(nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, proj_channels)
        # 3. InfoNCE 的灵魂：可学习的温度系数 (Temperature)
        # 初始化为 ln(1/0.07) = 2.6592，这是 CLIP 论文的标配初始值
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, feature_map):
        """
        Args:
            feature_map (Tensor): 主干网络吐出的特征图, 形状为 [B, in_channels, H, W]
        Returns:
            img_embeds (Tensor): 映射并归一化后的视觉向量, 形状为 [B, out_channels]
        """
        B, C, H, W = feature_map.shape
        x = self.gap(feature_map).view(B, C)
        x = self.proj(x)
        img_embeds = F.normalize(x, dim=-1)
        return img_embeds