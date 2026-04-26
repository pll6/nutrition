# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
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

    def pre_eval(self, preds, indices):
        """将模型输出的字典解包，适配分割评估，同时保留回归预测值。"""
        clean_preds = []
        n_preds = []
        
        # 1. 拆分：提取分割图和营养预测值
        for p in preds:
            if isinstance(p, dict):
                clean_preds.append(p.get('seg_map', p))
                n_preds.append(p.get('n_pred', None))
            else:
                clean_preds.append(p)
                n_preds.append(None)
                
        # ========================================================
        # 🐞 终极探针：截获第一张图的预测值与真值
        # ========================================================
        if not hasattr(self, '_printed_gt_debug'):
            import numpy as np
            print("\n" + "="*50)
            print("🐞 [GT vs Pred DEBUG INFO] 数据集交集探针：")
            
            pred_0 = clean_preds[0]
            print(f"1. 预测图 (Pred) shape: {pred_0.shape}")
            print(f"   -> 包含的像素类别: {np.unique(pred_0)}")
            
            gt_0 = self.get_gt_seg_map_by_idx(indices[0])
            print(f"2. 真值图 (GT) shape: {gt_0.shape}")
            print(f"   -> 包含的像素类别: {np.unique(gt_0)}")
            
            # 算一下两者的真实重合度
            pred_fg = (pred_0 == 1)
            gt_fg = (gt_0 == 1) 
            if gt_fg.sum() > 0:
                iou = (pred_fg & gt_fg).sum() / (pred_fg | gt_fg).sum()
                print(f"   -> 探针截获的单张图前景 IoU: {iou:.4f}")
            else:
                print("   -> ⚠️ 警告：GT 中没有类别 1！(可能你的真值把 255 当作了前景)")
            print("="*50 + "\n")
            self._printed_gt_debug = True
        # ========================================================

        # 2. 让父类处理分割部分
        seg_pre_evals = super().pre_eval(clean_preds, indices)
        
        # 3. 重新打包
        combined_results = []
        for seg_res, n_pred in zip(seg_pre_evals, n_preds):
            combined_results.append({
                'seg_pre_eval': seg_res,
                'n_pred': n_pred
            })
            
        return combined_results

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        import numpy as np
        import os.path as osp
        import re
        from collections import OrderedDict

        print("\n" + "*"*50)
        print("DEBUG: 正在进入 Evaluate 函数计算指标...")
        print("*"*50)

        metrics = metric if isinstance(metric, list) else [metric]
        
        # ========================================================
        # 1. 先拆解结果，把分割的部分还给父类去算 mIoU
        # ========================================================
        # results 此时是我们 pre_eval 打包的 dict 列表
        seg_results = [res['seg_pre_eval'] for res in results]
        
        # 过滤掉 'PMAE'，防止父类不认识这个指标报错
        seg_metrics = [m for m in metrics if m != 'PMAE']
        
        # 调用父类，拿到 mIoU 等分割指标
        if len(seg_metrics) > 0:
            eval_results = super().evaluate(seg_results, metric=seg_metrics, logger=logger, **kwargs)
        else:
            eval_results = OrderedDict()

        # ========================================================
        # 2. 我们自己计算 PMAE (营养素误差)
        # ========================================================
        if 'PMAE' in metrics or True: 
            csv_path = '/data/zengyuzhi/project/nutrition/data/ingredients/dish_metadata.csv'
            nutrition_dict = {} 
            
            if osp.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 6: continue
                        dish_id = parts[0]
                        macro_nutrients = np.array([
                            float(parts[1]), float(parts[2]), float(parts[3]), 
                            float(parts[4]), float(parts[5])
                        ], dtype=np.float32)
                        nutrition_dict[dish_id] = macro_nutrients
            else:
                print(f"\n[Error] 找不到营养标签文件: {csv_path}，请检查路径！")

            # 🚨 修改点 1：我们需要收集所有预测值和真值，而不是中途算误差
            all_preds = []
            all_gts = []
            
            for i, res in enumerate(results):
                n_pred_val = res.get('n_pred')
                if n_pred_val is None: 
                    continue
                
                pred_nutrition = np.array(n_pred_val).flatten()
                filename = self.img_infos[i].get('filename', '')
                match = re.search(r'(dish_\d+)', filename)
                
                if match and match.group(1) in nutrition_dict:
                    gt_nutrition = nutrition_dict[match.group(1)]
                    
                    # 收集所有的 pred 和 gt
                    all_preds.append(pred_nutrition)
                    all_gts.append(gt_nutrition)

            if len(all_preds) > 0:
                all_preds = np.array(all_preds) # [N, 5]
                all_gts = np.array(all_gts)     # [N, 5]
                
                # 🚨 修改点 2：严格按照论文公式 6 计算 PMAE
                # 公式 (5): 计算每个成分的 MAE (平均绝对误差)
                mae_k = np.mean(np.abs(all_preds - all_gts), axis=0)
                
                # 公式 (6) 的分母: 计算每个成分的 GT 平均值
                mean_gt_k = np.mean(all_gts, axis=0)
                
                # 为了防止除以 0，加一个小常数 epsilon
                mean_pmae = (mae_k / (mean_gt_k + 1e-6)) * 100
                
                nutrition_names = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
                
                # 把营养结果塞进 eval_results
                for name, val in zip(nutrition_names, mean_pmae):
                    eval_results[f'PMAE_{name}'] = round(val, 2)
                eval_results['mPMAE'] = round(np.mean(mean_pmae), 2)
                
                print("\n" + "="*40)
                print("        营养回归评价结果 (PMAE)      ")
                print("-" * 40)
                for k, v in eval_results.items():
                    if 'PMAE' in k:
                        print(f" {k:<15} : {v:>10}%")
                print("="*40 + "\n")
            else:
                print("\n[Warning] 匹配不到任何营养真值，请检查 CSV 或文件名！")

        return eval_results