import torch
import torch.nn as nn
from typing import List, Dict, Optional

from mmcv.runner import BaseModule
from mmcv.cnn import normal_init, constant_init
from mmseg.models.builder import HEADS, build_loss

@HEADS.register_module()
class MultiScaleNutritionHead(BaseModule):
    def __init__(self, 
                 in_channels_list: List[int], 
                 plate_embed_dim: int,
                 out_channels: int = 5, 
                 dropout_ratio: float = 0.2,
                 loss_reg: Dict = dict(type='LogL1Loss', loss_weight=1.0),
                 init_cfg: Optional[Dict] = None):

        super(MultiScaleNutritionHead, self).__init__(init_cfg)
        assert plate_embed_dim == in_channels_list[-1], \
            f"plate_embed_dim ({plate_embed_dim}) does not match the last element of in_channels_list ({in_channels_list[-1]})"

        self.in_channels_list = in_channels_list
        total_in_channels = sum(in_channels_list) + plate_embed_dim
        
        # 使用 AdaptiveAvgPool2d 确保无论输入尺寸多少，输出都是 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 构建 MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_in_channels, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(512, out_channels),
            nn.Softplus()  # 强制输出正数
        )
        
        self.loss_reg = build_loss(loss_reg)

    def init_weights(self):
        super().init_weights()
        if self.init_cfg is None:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    normal_init(m, mean=0, std=0.01)
                    if m.bias is not None:
                        constant_init(m.bias, 0)

    def forward(self, plate_embed: torch.Tensor, masked_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns:
            Tensor: 预测的营养数值, shape [B, out_channels].
        """

        assert len(masked_feats) == len(self.in_channels_list), \
            f"输入特征的数量 ({len(masked_feats)}) 与配置通道的长度 ({len(self.in_channels_list)}) 不匹配！"

        feats = []
        for feat in masked_feats:
            # pooled = self.gap(feat).flatten(1) 
            # 改用 Sum Pooling，保留食物在图中的绝对面积信息！面积越大，特征的绝对数值越大，物理逻辑才守恒
            pooled = feat.sum(dim=(2, 3))
            feats.append(pooled)
            
        feats = torch.cat(feats, dim=1) 
        feats = torch.cat([feats, plate_embed], dim=1)
        pred = self.mlp(feats)
        
        return pred

    def losses(self, pred: torch.Tensor, gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()
        losses['loss_reg'] = self.loss_reg(pred, gt_nutrition)
        return losses

    def forward_train(self, plate_embed: torch.Tensor, masked_feats: List[torch.Tensor], gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.forward(plate_embed, masked_feats)
        return self.losses(pred, gt_nutrition)