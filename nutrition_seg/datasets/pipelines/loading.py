# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import LoadImageFromFile

@PIPELINES.register_module()
class LoadImageWithDepthFromFile(LoadImageFromFile):
    def __init__(self, multiple_camara_param = True, **kwargs):
        super().__init__(**kwargs)

        self.multiple_camara_param = multiple_camara_param

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

        valid_pixels = depth_img[depth_img > 0]
        
        if self.multiple_camara_param:
            if len(valid_pixels) > 0:
                median_depth = np.median(valid_pixels)
                # 如果中位数大于 1500，说明用了高精度刻度 (0.1mm)，强行除以 10 统一为 1mm
                if median_depth > 1500:
                    depth_img = depth_img / 10.0

        results['depth'] = depth_img 

        if 'seg_fields' not in results:
            results['seg_fields'] = []
            
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
        