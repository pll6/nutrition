# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import LoadImageFromFile

@PIPELINES.register_module()
class LoadImageWithDepthFromFile(LoadImageFromFile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, results):
        results = super().__call__(results)

        depth_filename = results['img_info'].get('depth_filename')
        if depth_filename is None:
            raise KeyError("未在 results['img_info'] 中找到 'depth_filename'！"
                           "请检查 Dataset 的 prepare_train_img 是否正确注入了该字段。")

        img_bytes = self.file_client.get(depth_filename)

        # 5. 🚨 核心防坑：16位深度图防损解码
        # flag='unchanged' 等效于 cv2.IMREAD_UNCHANGED。
        # 绝对不能用默认的 'color'，否则底层会强行把单通道 16 位压缩成 3 通道 8 位，深度信息当场报废！
        depth_img = mmcv.imfrombytes(
            img_bytes, flag='unchanged', backend=self.imdecode_backend)

        depth_img = depth_img.astype(np.float32)

        if len(depth_img.shape) == 2:
            depth_img = np.expand_dims(depth_img, axis=-1)
        results['depth'] = depth_img

        assert 'seg_fields' in results
        if 'depth' not in results['seg_fields']:
            results['seg_fields'].append('depth')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"depth_flag='unchanged')"
        return repr_str
        