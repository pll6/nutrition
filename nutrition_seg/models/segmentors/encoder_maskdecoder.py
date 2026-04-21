# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize


@SEGMENTORS.register_module()
class EncoderMaskDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 pretrained,
                 alpha_dice=0.1,
                 regression_head=None,                 
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(EncoderMaskDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        decode_head.update(train_cfg=train_cfg)
        decode_head.update(test_cfg=test_cfg)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.regression_head = regression_head
        if regression_head is not None:
            self.regression_head = builder.build_head(regression_head)

        self.alpha_dice = alpha_dice

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, img, depth, *args, **kwargs):
        # 🚨 核心修复：如果是列表，取第一个元素 (DataContainer 的常见行为)
        if isinstance(depth, list):
            depth = depth[0]
        """Extract features from images."""
        x = self.backbone(img, depth, *args, **kwargs)
        if self.with_neck:
            x = self.neck(x, *args, **kwargs)
        return x

    def encode_decode(self, img, img_metas, depth=None):
        x = self.extract_feat(img, depth=depth)
        out = self._decode_head_forward_test(x, img_metas)
        
        # 🚨 这里的处理很关键
        if isinstance(out, tuple):
            # 假设 out = (cls_scores, mask_preds, plate_embed)
            cls_scores, mask_preds = out 
            
            # 模仿 MaskFormer 的推理逻辑：将 query 映射回像素空间
            cls_prob = cls_scores.softmax(-1)
            # 这里的索引 [:, :, :-1] 是为了去掉背景类，具体看你 num_classes 的定义
            seg_logit = torch.einsum('bqc,bqhw->bchw', cls_prob[:, :, :-1], mask_preds)
        else:
            seg_logit = out

        # 这样 resize 拿到的才是 [B, C, H, W] 的标准分割图
        out_resized = resize(
            input=seg_logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return out_resized

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, gt_labels, gt_masks, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        decoder_train_output = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, gt_labels, gt_masks, **kwargs)

        loss_decode = decoder_train_output.pop('loss')
        """
            {
            'decode.loss_cls': tensor(1.2345),
            'decode.loss_mask': tensor(0.8765),
            'decode.loss_dice': tensor(0.1234)
            }
        """
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, decoder_train_output

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img,
                      img_metas,
                      depth, 
                      gt_semantic_seg,
                      gt_labels,
                      gt_masks, 
                      gt_nutrition,
                      **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img, depth=depth)
        plate_embed, f1, f2, f3, f4 = x
        feats = [f1, f2, f3, f4]

        losses = dict()

        loss_decode, decoder_train_output = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, gt_labels, gt_masks, **kwargs)
        
        if self.regression_head is None:
            losses.update(loss_decode)
            return losses

        for k, v in loss_decode.items():
            losses[f'stage1_{k}'] = v * getattr(self, 'alpha_dice', 0.1)

        cls_scores = decoder_train_output['cls_scores'] # [B, num_queries, num_classes + 1]
        cls_prob = cls_scores.softmax(dim=-1) 
        fg_prob = cls_prob[..., 0] 

        mask_preds = decoder_train_output['mask_preds'] # [B, num_queries, H, W]
        mask_probs = mask_preds.sigmoid()
        weighted_mask_probs = mask_probs * fg_prob.unsqueeze(-1).unsqueeze(-1) # [B, Q, H, W]
        mask = torch.max(weighted_mask_probs, dim=1)[0].unsqueeze(1)

        masked_feats = []
        mask_h, mask_w = mask.shape[2], mask.shape[3]
        for f in feats:
            feat_h, feat_w = f.shape[2], f.shape[3]
            if feat_h == mask_h and feat_w == mask_w:
                mask_resized = mask
            
            else:
                mask_resized = F.interpolate(
                    mask, 
                    size=(feat_h, feat_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            masked_feat = f * mask_resized
            masked_feats.append(masked_feat)

        prob_sum = fg_prob.sum(dim=1, keepdim=True) + 1e-6
        plate_embed = torch.sum(fg_prob.unsqueeze(-1) * plate_embed, dim=1) / prob_sum

        loss_reg = self.regression_head.forward_train(plate_embed, masked_feats, gt_nutrition)
        losses.update(loss_reg)

        return losses
    
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, depth=None, **kwargs):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, depth)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, depth=None, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, depth=depth, **kwargs)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, depth=depth, **kwargs)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, depth=None, rescale=True, **kwargs):
        """Simple test with single image for Multi-task (Seg + Nutrition)."""
        
        # 1. 运行原有的分割推理获取 mask (注意传入了 depth)
        seg_logit = self.inference(img, img_meta, rescale, depth=depth)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy() # [B, H, W]

        # 如果没有回归头，直接返回分割结果
        if self.regression_head is None:
            return list(seg_pred)

        # 2. 手动提取回归所需的特征
        # 🚨 这里直接解包 List，确保后续不报错
        if isinstance(depth, list):
            depth = depth[0]
            
        x = self.extract_feat(img, depth=depth)
        plate_embed_raw, f1, f2, f3, f4 = x
        feats = [f1, f2, f3, f4]

        # 调用 decode_head 拿到分类分数和预测 mask
        decoder_output = self.decode_head(x, img_meta) 
        
        # 安全解包：确保无论是标准 Mask2Former 还是你自己的 MLPMaskDecoder 都能接住
        if isinstance(decoder_output, tuple):
            if len(decoder_output) == 2 and decoder_output[0].dim() == 3:
                cls_scores, mask_preds = decoder_output
            else:
                cls_scores = decoder_output[0][-1]  
                mask_preds = decoder_output[1][-1]  
        elif isinstance(decoder_output, dict):
            cls_scores = decoder_output['cls_scores']
            mask_preds = decoder_output['mask_preds']
        else:
            cls_scores, mask_preds = decoder_output

        # 3. 计算前景概率和掩码融合 (复刻训练时逻辑)
        cls_prob = cls_scores.softmax(dim=-1) 
        fg_prob = cls_prob[..., 0] 

        mask_probs = mask_preds.sigmoid()
        
        # --- 维度防御区 ---
        if fg_prob.dim() == 1:
            fg_prob = fg_prob.unsqueeze(0)
        if mask_probs.dim() == 3:
            mask_probs = mask_probs.unsqueeze(0)
            
        weighted_mask_probs = mask_probs * fg_prob.unsqueeze(-1).unsqueeze(-1) # [B, Q, H, W]
        mask = torch.max(weighted_mask_probs, dim=1)[0].unsqueeze(1)

        masked_feats = []
        mask_h, mask_w = mask.shape[2], mask.shape[3]
        for f in feats:
            feat_h, feat_w = f.shape[2], f.shape[3]
            if feat_h == mask_h and feat_w == mask_w:
                mask_resized = mask
            else:
                mask_resized = F.interpolate(
                    mask, 
                    size=(feat_h, feat_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            masked_feat = f * mask_resized
            masked_feats.append(masked_feat)

        # 🚨 关键修改区：直接使用原始的 1024 维 plate_embed_raw
        # 补齐 Batch 维度，防御 einsum 报错
        if plate_embed_raw.dim() == 2: 
            plate_embed_raw = plate_embed_raw.unsqueeze(0)
            
        # 聚合最终的盘子特征 (保持 1024 维，与回归头完美对齐)
        prob_sum = fg_prob.sum(dim=1, keepdim=True) + 1e-6
        plate_embed_final = torch.sum(fg_prob.unsqueeze(-1) * plate_embed_raw, dim=1) / prob_sum

        # 4. 回归头推理
        if hasattr(self.regression_head, 'forward_test'):
            n_pred = self.regression_head.forward_test(plate_embed_final, masked_feats)
        else:
            n_pred = self.regression_head(plate_embed_final, masked_feats)
        
        n_pred = n_pred.cpu().numpy() # [B, 5]

        # 5. 打包返回 [ {seg_map: mask, n_pred: pred}, ... ]
        results = []
        for i in range(len(seg_pred)):
            results.append(dict(
                seg_map=seg_pred[i], 
                n_pred=n_pred[i]
            ))
            
        return results

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`."""
        
        seg_result = None
        
        # 1. 灵活提取 seg_map
        if isinstance(result, dict) and 'seg_map' in result:
            seg_result = result['seg_map']
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'seg_map' in result[0]:
                seg_result = result[0]['seg_map']
            else:
                seg_result = result[0]
        else:
            seg_result = result

        # 2. 🚨 核心修复：强制维度对齐
        import numpy as np
        if isinstance(seg_result, torch.Tensor):
            seg_result = seg_result.detach().cpu().numpy()
        
        # 如果是 (1, H, W) 或 (B, H, W)，压缩掉多余的 1
        if isinstance(seg_result, np.ndarray):
            if seg_result.ndim == 3 and seg_result.shape[0] == 1:
                seg_result = seg_result.squeeze(0)
            elif seg_result.ndim == 3 and seg_result.shape[-1] == 1:
                seg_result = seg_result.squeeze(-1)
            
            # 最后的防线：如果还是不是 2D，强行取第一层
            if seg_result.ndim != 2:
                # 这可能是因为 batch 维度没压掉，或者有别的通道
                while seg_result.ndim > 2:
                    seg_result = seg_result[0]

        # 3. 调用父类
        return super(EncoderMaskDecoder, self).show_result(
            img,
            seg_result,
            palette=palette,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            opacity=opacity)