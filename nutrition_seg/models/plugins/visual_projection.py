import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List

from mmcv.runner import BaseModule
from mmcv.cnn import normal_init, constant_init
from mmseg.models.builder import LOSSES
from ..builder import TEXT_ENCODERS

@TEXT_ENCODERS.register_module()
class VisualProjectionBranch(BaseModule):
    def __init__(self, 
                 in_channels,
                 proj_channels: int = 512,
                 loss_clip: Optional[Dict] = None,
                 loss_weight: float = 0.1,     # 引入辅助权重，保护回归任务
                 init_cfg: Optional[Dict] = None):
        super(VisualProjectionBranch, self).__init__(init_cfg)
        
        self.loss_weight = loss_weight
        self.proj_channels = proj_channels
        
        # 全局平均池化，用于把每个尺度的 [B, C, H, W] 压成 1D 向量
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 拼接后的总通道数
        total_channels = in_channels * 4
        
        # 特征解耦防线：洗掉回归任务所需的绝对数值，只留供对比学习使用的相对语义
        self.semantic_norm = nn.LayerNorm(total_channels)
        
        # 融合与降维的 MLP
        self.proj = nn.Sequential(
            nn.Linear(total_channels, total_channels // 2),
            nn.GELU(),
            nn.Linear(total_channels // 2, proj_channels)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if loss_clip is not None:
            self.loss_clip = LOSSES.build(loss_clip)
        else:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
            else:
                rank, world_size = 0, 1
                
            if world_size > 1:
                self.loss_clip = LOSSES.build(
                    dict(type='ClipLoss', local_loss=False, gather_with_grad=True, rank=rank, world_size=world_size))
            else:
                self.loss_clip = LOSSES.build(   
                    dict(type='SingleGPUCrossBatchLoss', feature_dim=proj_channels, queue_size=1024))

    def init_weights(self):
        super().init_weights()
        if self.init_cfg is None:
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    normal_init(m, mean=0, std=0.01)
                    if m.bias is not None:
                        constant_init(m.bias, 0)

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        🚨 核心修改：这里完美接住传进来的 feats 列表 [f1, f2, f3, f4]
        """
        pooled_feats = []
        for feat in feature_maps:
            B, C = feat.shape[0], feat.shape[1]
            # 把每个特征图池化成 [B, C] 的向量
            pooled = self.gap(feat).view(B, C)
            pooled_feats.append(pooled)
            
        x = torch.cat(pooled_feats, dim=1)
        
        # 解耦归一化 -> 降维 -> L2归一化对齐 Text
        x = self.semantic_norm(x)
        x = self.proj(x)
        img_embeds = F.normalize(x, dim=-1)
        
        return img_embeds

    def forward_train(self, feature_maps: List[torch.Tensor], text_embeds: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        img_embeds = self.forward(feature_maps)
        losses = dict()
        if self.loss_clip is not None and text_embeds is not None:
            raw_loss = self.loss_clip(img_embeds, text_embeds, self.logit_scale, **kwargs)
            losses['loss_clip'] = raw_loss * self.loss_weight
        return losses