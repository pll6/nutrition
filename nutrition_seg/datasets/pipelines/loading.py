# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
import numpy as np

import mmcv
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import LoadImageFromFile

@PIPELINES.register_module()
class LoadImageWithDepthFromFile(LoadImageFromFile):
    def __init__(self, multiple_camera_param = True, **kwargs):
        super().__init__(**kwargs)

        self.multiple_camera_param = multiple_camera_param

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
        
        if self.multiple_camera_param:
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



@PIPELINES.register_module()
class LoadNutritionFromCSV(object):
    """从 Nutrition5k 的 CSV 文件中读取营养标签和食材文本
    
    Args:
        csv_path (str): dish_metadata.csv 的绝对路径
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # 初始化时在内存中建好哈希索引，避免训练时重复读取文件
        self.nutrition_dict = self._parse_csv(csv_path)

    def _parse_csv(self, csv_path):
        nutrition_data = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                    
                dish_id = parts[0]
                
                # 提取 5 维营养回归目标: Calories, Mass, Fat, Carb, Protein
                # 统一转换为 np.float32，这是 OpenMMLab 底层转 Tensor 的标准格式
                macro_nutrients = np.array([
                    float(parts[1]), float(parts[2]), float(parts[3]), 
                    float(parts[4]), float(parts[5])
                ], dtype=np.float32)
                
                # 提取变长的食材名称列表 (为后续输入 FLAVA/CLIP 做准备)
                ingr_names = []
                ingr_data = parts[6:]
                # 每 7 个字段为一组食材数据，索引 1 是食材名称
                for i in range(0, len(ingr_data), 7):
                    if i + 1 < len(ingr_data):
                        ingr_names.append(ingr_data[i+1])
                
                # 存入字典
                nutrition_data[dish_id] = {
                    'nutrition_vector': macro_nutrients,
                    'ingredient_names': ingr_names
                }
        return nutrition_data

    def __call__(self, results):
        # 1. 尝试获取文件名
        filename = results['img_info'].get('filename', '')
        
        # 2. 提取 dish_id (假设你的图片名类似 'dish_1561662216_rgb.png')
        # 正则匹配 'dish_' 加上后面的一串数字
        match = re.search(r'(dish_\d+)', filename)
        
        if not match:
            raise ValueError(f"无法从文件名 {filename} 中提取出 'dish_id' 格式的标识符！")
            
        dish_id = match.group(1)
        
        if dish_id not in self.nutrition_dict:
            raise KeyError(f"在 CSV 文件中找不到对应的菜品 ID: {dish_id}")
            
        # 3. O(1) 极速查询字典，并将数据注入到 results 中
        dish_info = self.nutrition_dict[dish_id]
        
        # 存入回归 Ground Truth (shape: [5])
        results['gt_nutrition'] = dish_info['nutrition_vector']
        
        # 存入文本 Ground Truth (List[str])
        # 例如: ['soy sauce', 'garlic', 'white rice', ...]
        results['gt_ingredients'] = dish_info['ingredient_names']
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(csv_path={self.csv_path}, loaded_dishes={len(self.nutrition_dict)})'
        return repr_str