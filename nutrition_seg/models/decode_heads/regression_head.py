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
                 normalize: bool,
                 loss_reg,
                 train_means=None,
                 train_std=None,
                 out_channels: int = 5, 
                 dropout_ratio: float = 0.2,
                 init_cfg: Optional[Dict] = None):

        super(MultiScaleNutritionHead, self).__init__(init_cfg)
        assert plate_embed_dim == in_channels_list[-1], \
            f"plate_embed_dim ({plate_embed_dim}) does not match the last element of in_channels_list ({in_channels_list[-1]})"

        self.in_channels_list = in_channels_list
        self.plate_embed_dim = plate_embed_dim
        self.out_channels = out_channels
        
        food_channels = sum(in_channels_list)
        

        # 1. 🚨 第一道防线：驯服 Sum Pooling 产生的庞大数值
        # self.food_norm = nn.LayerNorm(food_channels)
        self.plate_norm = nn.LayerNorm(plate_embed_dim)
        
        self.plate_modulator = nn.Sequential(
            nn.Linear(plate_embed_dim, food_channels // 2),
            nn.GELU(),
            nn.Linear(food_channels // 2, food_channels),
            nn.Sigmoid()
        )

        total_in_channels = food_channels + plate_embed_dim
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

        softplus = nn.Softplus()
        self.normalize = normalize
        if self.normalize:
            assert train_means is not None and train_std is not None, "如果 normalize=True，必须提供 train_means 和 train_std！"
            self.register_buffer('nutrition_means', torch.tensor(train_means).float().reshape(1, -1))
            self.register_buffer('nutrition_std', torch.tensor(train_std).float().reshape(1, -1))
            softplus = nn.Identity()

        self.softplus = softplus
        
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
            
        # 2. 独立归一化 (将巨大的 Sum 面积压缩到健康分布，同时保留相对大小)
        food_feats = torch.cat(feats, dim=1)
        food_feats = food_feats / 10000.0
        
        plate_embed = self.plate_norm(plate_embed)
        
        # 3. 门控融合：用盘子作为参照物，计算物理“体积/密度”
        plate_gate = self.plate_modulator(plate_embed)
        modulated_food = food_feats * plate_gate
        
        # 4. 最终预测
        feats = torch.cat([modulated_food, plate_embed], dim=1)
        raw_logits = self.mlp(feats)
        pred = self.softplus(raw_logits)

        # ==========================================
        # 🐞 核心 DEBUG 监控区（每隔一定步数打印一次）
        # ==========================================
        # if self.training and torch.rand(1).item() < 0.01: # 大约 5% 的概率触发打印，防刷屏
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
        
        # ========================================================
        # 🧪 验证集探针：Plate 特征敏感度消融实验 (已修复归一化 Bug)
        # ========================================================
        if not self.training:
            if torch.rand(1).item() < 0.05:
                fake_plate = torch.zeros_like(plate_embed)
                fake_plate_normed = self.plate_norm(fake_plate)
                fake_gate = self.plate_modulator(fake_plate_normed)
                fake_modulated = food_feats * fake_gate
                fake_feats_cat = torch.cat([fake_modulated, fake_plate_normed], dim=1)
                fake_pred = self.softplus(self.mlp(fake_feats_cat))
                
                # 🚨 修复：在算百分比偏离度之前，先局部转换回物理量纲，防止分母为负数
                probe_real_pred = pred
                probe_fake_pred = fake_pred
                if self.normalize:
                    probe_real_pred = pred * self.nutrition_std + self.nutrition_means
                    probe_fake_pred = fake_pred * self.nutrition_std + self.nutrition_means
                
                # 保证分母为正，避免负数百分比
                probe_real_pred = torch.clamp(probe_real_pred, min=1e-3)
                probe_fake_pred = torch.clamp(probe_fake_pred, min=0.0)

                deviation = torch.abs(probe_real_pred - probe_fake_pred) / probe_real_pred
                mean_dev = deviation.mean().item() * 100
                
                print("\n" + "🧪" * 25)
                print(f"[验证集 DEBUG] Plate 门控消融探针:")
                print(f" -> 正常预测平均值(物理尺度): {probe_real_pred.mean().item():.2f}")
                print(f" -> 抹除盘子后，预测值平均偏离了: {mean_dev:.2f}%")
                if mean_dev < 1.0:
                    print(" 🚨 灾难警告：门控失效！模型依然没看盘子！")
                elif mean_dev > 10.0:
                    print(" ✅ 健康：门控生效！模型极度依赖盘子尺度信息！")
                print("🧪" * 25 + "\n")
        # ========================================================

        if not self.training and self.normalize:
            pred = pred * self.nutrition_std + self.nutrition_means
            pred = torch.clamp(pred, min=0.0)

        return pred

    def losses(self, pred: torch.Tensor, gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()

        if self.normalize:
            # 归一化标签，确保数值范围匹配
            gt_nutrition = (gt_nutrition - self.nutrition_means) / (self.nutrition_std + 1e-6)

        losses['loss_reg'] = self.loss_reg(pred, gt_nutrition)
        return losses

    def forward_train(self, plate_embed: torch.Tensor, masked_feats: List[torch.Tensor], gt_nutrition: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.forward(plate_embed, masked_feats, gt_nutrition)
        return self.losses(pred, gt_nutrition)





        