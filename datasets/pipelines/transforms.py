# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import Pad

@PIPELINES.register_module()
class RandomForegroundCrop(object):
    """专为二值分割定制的前景裁剪机"""
    def __init__(self, crop_size, foreground_id=1, min_pixels=10):
        self.crop_size = crop_size
        self.foreground_id = foreground_id  # 前景的类别序号
        self.min_pixels = min_pixels        # 只要框里的前景像素大于这个数，就算好图

    def get_crop_bbox(self, img):
        # ... (和原来盲人摸象的坐标逻辑完全一样) ...
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_h + self.crop_size[0], offset_w, offset_w + self.crop_size[1]

    def crop(self, img, crop_bbox):
        y1, y2, x1, x2 = crop_bbox
        return img[y1:y2, x1:x2, ...]

    def __call__(self, results):
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        
        # 🚨 核心改动：二值分割专属判定
        for _ in range(10):
            seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
            
            # 数一数切片里，前景像素有几个
            fg_pixels = np.sum(seg_temp == self.foreground_id)
            
            # 如果前景像素够多，说明构图不错，直接跳出！
            if fg_pixels >= self.min_pixels:
                break
                
            crop_bbox = self.get_crop_bbox(img) # 没切到前景，重新切！

        # 执行真正的裁剪
        results['img'] = self.crop(img, crop_bbox)
        results['img_shape'] = results['img'].shape
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results
    
    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class ModifiedPad(Pad):
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        super().__init__(size, size_divisor, pad_val, seg_pad_val)

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            if key == 'depth':  
                results[key] = mmcv.impad(
                    results[key],
                    shape=results['pad_shape'][:2],
                    pad_val=-1) 
            else:
                results[key] = mmcv.impad(
                    results[key],
                    shape=results['pad_shape'][:2],
                    pad_val=self.seg_pad_val)       

@PIPELINES.register_module()
class NormalizeDepth(object):
    def __init__(self, mode=None, mean=None, std=None, max_depth=None):
        self.mode = mode
        self.mean = mean
        self.std = std
        self.max_depth = max_depth

    def __call__(self, results):
        depth = results['depth']
 
        if self.mode == 'z-score':
            if self.mean is None or self.std is None:
                raise ValueError(" NormalizeDepth 报错：使用 'z-score' 模式时，必须在 config 中提供 mean 和 std 参数！")
   
            depth = (depth - self.mean) / self.std
            
        elif self.mode == 'min-max':
            if self.max_depth is None:
                raise ValueError(" NormalizeDepth 报错：使用 'min-max' 模式时，必须在 config 中提供 max_depth 参数！")
      
            depth = depth / self.max_depth
            depth = np.clip(depth, 0.0, 1.0)
            
        else:
            raise ValueError(f" NormalizeDepth 报错：不支持的归一化模式 '{self.mode}'，请使用 'z-score' 或 'min-max'。")
        
        results['depth'] = depth
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        if self.mode == 'z-score':
            repr_str += f"(mode='{self.mode}', mean={self.mean}, std={self.std})"
        else:
            repr_str += f"(mode='{self.mode}', max_depth={self.max_depth})"
        return repr_str