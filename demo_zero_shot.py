import os
import torch
import csv
from tqdm import tqdm  # 引入进度条，方便看跑分进度
from mmcv.utils import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
import nutrition_seg

def load_gt_ingredients(csv_path):
    """提取真实标签映射 (修复了 Nutrition5k 横向扩展的问题)"""
    dish_to_ingr = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8: continue
                dish_id = row[0].strip()
                if dish_id not in dish_to_ingr:
                    dish_to_ingr[dish_id] = []
                for col_idx in range(7, len(row), 7):
                    ingr_name = row[col_idx].strip()
                    if ingr_name and ingr_name not in dish_to_ingr[dish_id]:
                        dish_to_ingr[dish_id].append(ingr_name)
        return dish_to_ingr
    except Exception as e:
        print(f"❌ 加载真值失败: {e}")
        return {}
    
def load_vocabulary_from_csv(csv_path):
    """加载测试词汇表"""
    vocabulary = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) >= 1: 
                    vocabulary.append(row[0].strip())
    except Exception as e:
        pass
    return vocabulary

def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    config_file = 'nutrition_seg/config/nutrition_segmentation/maskdecoder_vit_adapter_large_512_50k_CLIP_text_encoder_nutrition_segmentation_stage1.py'
    checkpoint_file = '/data/zengyuzhi/project/nutrition/work_dirs/CLIP_segment/TTFF/iter_30000.pth' # 评测时记得换成你想测的权重
    csv_ingredients_path = '/data/zengyuzhi/project/nutrition/data/ingredients/ingredients_metadata.csv'
    dish_metadata_path = '/data/zengyuzhi/project/nutrition/data/ingredients/dish_metadata.csv'

    print("=> 正在初始化评测环境...")
    test_vocabulary = load_vocabulary_from_csv(csv_ingredients_path)
    gt_mapping = load_gt_ingredients(dish_metadata_path)

    cfg = Config.fromfile(config_file)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_file, map_location='cuda:0')
    model.cuda()
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
    
    # ==========================================
    # 2. 初始化统计变量 (核心打分器)
    # ==========================================
    total_valid_images = 0
    hit_at_1 = 0
    hit_at_5 = 0

    print(f"\n🚀 开始全量验证集测试，共 {len(dataset)} 张图片...")
    
    # ==========================================
    # 3. 批量推理循环
    # ==========================================
    with torch.no_grad():
        # tqdm 会在终端画一个漂亮的进度条
        for data in tqdm(data_loader, desc="评测进度"):
            
            img_tensor = data['img'][0].cuda()
            img_meta = data['img_metas'][0].data[0]
            depth_tensor = data['depth'][0].cuda() if 'depth' in data else None
            
            raw_filename = os.path.basename(img_meta[0]['filename'])
            current_dish_id = os.path.splitext(raw_filename)[0]
            
            # 拿到真值列表，如果没有或者叫"未知"，则跳过该图不计入成绩
            gt_list = gt_mapping.get(current_dish_id, [])
            if not gt_list:
                continue
                
            total_valid_images += 1
            
            # 推理
            results = model.zero_shot_predict_multilabel(
                img=img_tensor, 
                img_meta=img_meta, 
                vocabulary=test_vocabulary, 
                depth=depth_tensor, 
                threshold=0.0  # 评测排序不需要阈值拦截
            )
            
            # 拿到全量得分字典并排序
            all_scores = results[0]['all_scores']
            sorted_preds = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 提取 Top-5 的纯单词列表
            top_5_words = [item[0] for item in sorted_preds[:5]]
            top_1_word = top_5_words[0]
            
            # 🎯 计算 Recall@1: 第一名是不是命中了任何一个真实标签？
            if top_1_word in gt_list:
                hit_at_1 += 1
                
            # 🎯 计算 Recall@5: 前五名里有没有命中任何一个真实标签？
            if any(word in gt_list for word in top_5_words):
                hit_at_5 += 1

    # ==========================================
    # 4. 输出最终成绩单
    # ==========================================
    if total_valid_images > 0:
        recall_1 = (hit_at_1 / total_valid_images) * 100
        recall_5 = (hit_at_5 / total_valid_images) * 100
        
        print("\n" + "="*50)
        print("🏆 零样本对比学习评估报告 (Zero-Shot Evaluation)")
        print("="*50)
        print(f"评测权重文件: {os.path.basename(checkpoint_file)}")
        print(f"有效评测样本: {total_valid_images} 张图像")
        print("-" * 50)
        print(f"🔥 Recall@1 (Top-1 命中率): {recall_1:.2f}%")
        print(f"🌟 Recall@5 (Top-5 命中率): {recall_5:.2f}%")
        print("="*50 + "\n")
    else:
        print("\n⚠️ 评测失败：没有在 CSV 中找到与测试集图片匹配的 GT 标签。")

if __name__ == '__main__':
    main()