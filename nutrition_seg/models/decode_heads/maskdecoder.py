from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.adapter_modules import FFN

from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ...core import multi_apply, reduce_mean
from ..builder import build_assigner

@HEADS.register_module()
class MLPMaskDecoder(BaseDecodeHead):
    def __init__(self, in_features, hidden_features, out_features, num_plate_emd, num_classes=1, mlp_num_layers=3,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_mask=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 assigner=dict(
                     type='MaskHungarianAssigner',
                     cls_cost=dict(type='ClassificationCost', weight=1.),
                     dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True,
                                    eps=1.0),
                     mask_cost=dict(type='MaskFocalLossCost', weight=20.0)),
                 **kwargs):
        
        kwargs.pop('train_cfg', None)
        kwargs.pop('test_cfg', None)

        super().__init__(in_channels=in_features, channels=1, num_classes=num_classes, **kwargs) # 用不上，防止报错，无实际意义
        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        
        self.plate_mlp = FFN(in_features, hidden_features, out_features, mlp_num_layers, drop=0)
        self.feat_mlp = FFN(
            in_features,
            hidden_features,
            out_features,
            mlp_num_layers,
            affine_func=dense_affine_func,
            drop=0
        )
        self.class_embed = nn.Linear(out_features, num_classes + 1)

        self.num_queries = num_plate_emd
        self.assigner = build_assigner(assigner)

        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__.__name__ == 'MLPMaskDecoder'):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official MaskFormerHead repo, bg_cls_weight
            # means relative classification weight of the VOID class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = (self.num_classes + 1) * [class_weight]
            # set VOID class as the last indice
            class_weight[self.num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            'The classification weight for loss and matcher should be' \
            'exactly the same.'
        assert loss_dice['loss_weight'] == assigner['dice_cost']['weight'], \
            f'The dice weight for loss and matcher' \
            f'should be exactly the same.'
        assert loss_mask['loss_weight'] == assigner['mask_cost']['weight'], \
            'The focal weight for loss and matcher should be' \
            'exactly the same.'
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.init_weights()

    def init_weights(self, m=None):
        nn.init.normal_(self.class_embed.weight, std=0.01)
        # 使用先验概率初始化 bias
        # 假设在训练初期，绝大多数 query 是背景，只有约 10% 能命中盘子
        prior_prob = 0.1
        
        # Softmax 下，设背景 bias 为 0，计算前景 bias
        # p = exp(b_fg) / (exp(b_fg) + exp(b_bg)) -> b_fg = ln(p / (1-p))
        bias_value = math.log(prior_prob / (1 - prior_prob))
        
        # 先把所有 bias 初始化为 0 (代表背景)
        nn.init.constant_(self.class_embed.bias, 0)
        
        # 将前景类 (索引 0 到 num_classes-1) 的 bias 设置为负数 (-2.19)
        # 这样模型初始预测前景的概率约为 10%，避免初期 Loss 爆炸
        nn.init.constant_(self.class_embed.bias[:self.num_classes], bias_value)

    def forward(self, feats, img_metas):
        """
        Args:
            feats (list[Tensor]): Features from the upstream network, each
                is a 4D-tensor(with plate_embed).
            img_metas (list[dict]): List of image information.

        Returns:
            cls_scores (Tensor): The class scores for each query, has shape (batch_size, num_queries, num_classes).
            mask_preds (Tensor): The mask predictions for each query, has shape (batch_size, num_queries, height, width).
        """
        plate_embed, f1, f2, f3, f4 = feats

        plate_embed = self.plate_mlp(plate_embed)
        f1 = self.feat_mlp(f1)
        bs, c, h, w = f1.shape

        # preidict mask
        mask_preds = torch.einsum("bqc,bchw->bqhw", plate_embed, f1)
        cls_scores = self.class_embed(plate_embed)
        return cls_scores, mask_preds

    def forward_train(self, x,
                      img_metas,
                      gt_semantic_seg,
                      gt_labels,
                      gt_masks):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """
        cls_scores, mask_preds = self(x, img_metas)

        return {'loss' : self.loss(cls_scores, mask_preds, gt_labels, gt_masks, img_metas),
                'cls_scores': cls_scores, 
                'mask_preds': mask_preds
                }
    
    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs, img_metas)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_queries,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_queries, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_queries, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_queries, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_queries, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)
    
    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape [num_queries, h, w].
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                    shape [num_queries, ].
                - label_weights (Tensor): Label weights of each image.
                    shape [num_queries, ].
                - mask_targets (Tensor): Mask targets of each image.
                    shape [num_queries, h, w].
                - mask_weights (Tensor): Mask weights of each image.
                    shape [num_queries, ].
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        gt_masks_downsampled = F.interpolate(
            gt_masks.unsqueeze(1).float(), target_shape,
            mode='nearest').squeeze(1).long()
        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels,
                                             gt_masks_downsampled, img_metas)
        # pos_ind: range from 1 to (self.num_classes)
        # which represents the positive index
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)

        # mask target
        mask_targets = gt_masks[pos_assigned_gt_inds, :]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape [batch_size, num_queries,
                cls_out_channels].
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape [batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]:Loss components for outputs from a single decoder
                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape [batch_size, num_queries]
        labels = torch.stack(labels_list, dim=0)
        # shape [batch_size, num_queries]
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape [num_gts, h, w]
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape [batch_size, num_queries]
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape [batch_size * num_queries, ]
        cls_scores = cls_scores.flatten(0, 1)
        # shape [batch_size * num_queries, ]
        labels = labels.flatten(0, 1)
        # shape [batch_size* num_queries, ]
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_ones(self.num_classes + 1)
        class_weight[-1] = self.bg_cls_weight

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # upsample to shape of target
        # shape [num_gts, h, w]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # FocalLoss support input of shape [n, num_class]
        h, w = mask_preds.shape[-2:]
        # shape [num_gts, h, w] -> [num_gts * h * w, 1]
        mask_preds = mask_preds.reshape(-1, 1)
        # shape [num_gts, h, w] -> [num_gts * h * w]
        mask_targets = mask_targets.reshape(-1)
 
        loss_mask = self.loss_mask(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_dice=loss_dice
        )