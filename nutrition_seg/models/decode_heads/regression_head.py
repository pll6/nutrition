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
        # 1. 🚨 第一道防线：驯服 Sum Pooling 产生的庞大数值
        self.feat_norm = nn.LayerNorm(total_in_channels)
        
        # 2. 🚨 换上不死激活函数 GELU，并给每一层加上护具 LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(total_in_channels, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout_ratio),
            
            nn.Linear(512, out_channels)
            # 注意：我把 Softplus 移到了 forward 里面，方便做 Debug 拦截！
        )
        
        self.softplus = nn.Softplus()
        
        self.loss_reg = build_loss(loss_reg)

    def init_weights(self):
        super().init_weights()
        if self.init_cfg is None:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    normal_init(m, mean=0, std=0.01)
                    if m.bias is not None:
                        constant_init(m.bias, 0)

    def forward(self, plate_embed: torch.Tensor, masked_feats: List[torch.Tensor], gt_nutrition: torch.Tensor = None) -> torch.Tensor:
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
        # 🚨 必须在送入 Linear 之前 Norm，否则数值爆炸
        feats = self.feat_norm(feats)
        
        # 拿到 Linear 的原始输出（还未经过 Softplus）
        raw_logits = self.mlp(feats)
        
        # 经过 Softplus 保证正数
        pred = self.softplus(raw_logits)

        # ==========================================
        # 🐞 核心 DEBUG 监控区（每隔一定步数打印一次）
        # ==========================================
        # if self.training and torch.rand(1).item() < 0.5: # 大约 5% 的概率触发打印，防刷屏
        #     print("\n" + "="*50)
        #     print("🚀 [DEBUG] 回归头生命体征监控:")
        #     print(f"1. 输入特征 (Sum Pooled): Mean={feats.mean().item():.4f}, Max={feats.max().item():.4f}, Min={feats.min().item():.4f}")
        #     print(f"2. Linear 原始输出 (Raw Logits): Mean={raw_logits.mean().item():.4f}, Max={raw_logits.max().item():.4f}, Min={raw_logits.min().item():.4f}")
        #     print(f"   >>> 如果这里全是 -20 以下的负数，说明你的网络被砸晕了，Softplus 的梯度已死！")
        #     print(f"3. Softplus 最终预测 (Pred): Mean={pred.mean().item():.4f}, Max={pred.max().item():.4f}")
        #     if gt_nutrition is not None:
        #         print(f"4. 真实标签 (GT Label): Mean={gt_nutrition.mean().item():.4f}, Max={gt_nutrition.max().item():.4f}")
        #         print(f"   >>> 如果 Label 是 1000，而 Pred 是 0.x，你需要考虑归一化 Label！")
        #     print("="*50 + "\n")
        
        return pred

    def losses(self, pred: torch.Tensor, gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()
        losses['loss_reg'] = self.loss_reg(pred, gt_nutrition)
        return losses

    def forward_train(self, plate_embed: torch.Tensor, masked_feats: List[torch.Tensor], gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.forward(plate_embed, masked_feats, gt_nutrition)
        return self.losses(pred, gt_nutrition)