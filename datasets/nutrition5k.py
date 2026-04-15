# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module(force=True)
class Nutrition5kDataset(CustomDataset):
    """
    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('background', 'foreground')

    PALETTE = [[255, 255, 255], [0, 0, 0]]

    def __init__(self, **kwargs):
        self.depth_img_dir = kwargs.pop('depth_img_dir', None)
        assert self.depth_img_dir is not None, "depth_img_dir is required for Nutrition5kDataset}"
        
        self.depth_img_suffix = '.png'
        
        # 此时 kwargs 里已经没有 depth_img_dir 了，父类安全！
        super(Nutrition5kDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

        if self.data_root is not None:
            if not (self.depth_img_dir is None or osp.isabs(self.depth_img_dir)):
                self.depth_img_dir = osp.join(self.data_root, self.depth_img_dir)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        
        # 拿掉 RGB 后缀，换上深度图后缀
        filename = img_info['filename']
        depth_filename = filename.replace(self.img_suffix, self.depth_img_suffix)
        
        # 这里的 self.depth_img_dir 已经在上面被完美处理成绝对路径了！
        if self.depth_img_dir is not None:
            img_info['depth_filename'] = osp.join(self.depth_img_dir, depth_filename)

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        
        filename = img_info['filename']
        depth_filename = filename.replace(self.img_suffix, self.depth_img_suffix)
        
        if self.depth_img_dir is not None:
            img_info['depth_filename'] = osp.join(self.depth_img_dir, depth_filename)

        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)