# =========================================================
# 1. 全局变量区 (参数一目了然，只在这里改)
# =========================================================
num_classes = 1 
num_queries = 20
crop_size = (512, 512)
depth_mean = 375.43  
depth_std = 89.56

# =========================================================
# 2. 数据处理区 (你亲手组装的无敌流水线)
# =========================================================
csv_path='/data/zengyuzhi/project/nutrition/data/ingredients/dish_metadata.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_means = [254.41, 214.82, 12.75, 19.12, 18.08]
train_std = [209.83, 153.77, 13.52, 15.99, 20.05]

train_pipeline = [
    dict(type='LoadImageWithDepthFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadNutritionFromCSV', csv_path=csv_path),
    dict(
        type='Resize', 
        img_scale=(512, 512),   
        ratio_range=None,   
        keep_ratio=True      
    ),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='NormalizeDepth', mode='z-score', mean=depth_mean, std=depth_std),
    dict(type='ModifiedPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth', 'gt_semantic_seg', 'gt_masks', 'gt_labels', 'gt_nutrition'])
]

test_pipeline = [
    dict(type='LoadImageWithDepthFromFile'),
    # 注意：测试集绝对不能做 Crop 和 Flip，也不能加载 Annotations
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),   
        flip=False,           
        transforms=[
            dict(type='Resize', keep_ratio=True), 
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormalizeDepth', mode='z-score', mean=depth_mean, std=depth_std),
            dict(type='ModifiedPad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img', 'depth']),
            dict(type='Collect', keys=['img', 'depth'])
        ])
]
# =========================================================
# 3. 数据集大本营区 (指定花名册)
# =========================================================
data = dict(
    samples_per_gpu=8,  
    workers_per_gpu=8,
    train=dict(
        type='Nutrition5kDataset',
        data_root='/data/zengyuzhi/project/nutrition/data',
        img_dir='images/rgb',
        ann_dir='segmentations',
        depth_img_dir='images/depth_raw',
        split='splits/train_split_20260416_1116.txt',
        pipeline=train_pipeline),
    val=dict(
        type='Nutrition5kDataset',
        # ... 同上 ...
        split='splits/val_split_20260416_1116.txt',
        pipeline=test_pipeline),
    test=dict(
        type='Nutrition5kDataset',
        data_root='/data/zengyuzhi/project/nutrition/data',
        img_dir='images/rgb',
        ann_dir='segmentations',
        depth_img_dir='images/depth_raw',
        split='splits/val_split_20260416_1116.txt',
        pipeline=test_pipeline)
)

# =========================================================
# 4. 核心模型区 (网络结构)
# =========================================================
model = dict(
    type='EncoderMaskDecoder',
    pretrained='pretrained/vit_large_patch16_224_augreg.pth', 
    backbone=dict(
        type='ViTAdapter',
        # img_size=512,
        patch_size=16,
        embed_dim=1024,   # <--- Base是768，Large必须是1024
        depth=24,     # <--- Base是12，Large必须是24
        num_heads=16,      # <--- Base是12，Large必须是16
        mlp_ratio=4,
        drop_path_rate=0.3,
        qkv_bias=True,
        init_values=1e-6,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        with_depth = [True, True, True, True], 
        num_plate_emd=num_queries,
    ),
    decode_head=dict(
        type='MLPMaskDecoder',
        in_features=1024,  
        hidden_features=256,
        out_features=256,
        num_classes=1,
        mlp_num_layers=3,
        num_plate_emd=num_queries,
    ),
    regression_head=dict(
        type='MultiScaleNutritionHead',
        in_channels_list=[1024, 1024, 1024, 1024],
        plate_embed_dim=1024,
        normalize=True,
        train_means=train_means,
        train_std=train_std,
        loss_reg=dict(type='SmoothL1Loss', loss_weight=20.0),
    ), 
    test_cfg=dict(mode='whole'),
    alpha_dice=0.5,
)

# =========================================================
# 5. 训练引擎配置区 (优化器、学习率、保存策略)
# =========================================================
# AdamW 是 ViT 和 MaskFormer 系列的标配
# optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.05, 
#                  paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
optimizer = dict(
    type='AdamW', 
    lr=0.0002,  
    weight_decay=0.05, 
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5),      # Backbone 稳住 (实际 LR = 1e-5)
            'decode_head': dict(lr_mult=0.1),   # 分割头 稳住 (实际 LR = 1e-5)
            'regression_head': dict(lr_mult=5.0) # 🚨 核心：回归头步长放大10倍！(实际 LR = 1e-3)
        }
    )
)
optimizer_config = dict()

# 学习率衰减策略
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False,
# 🚨 新增：线性预热机制
    warmup='linear',
    warmup_iters=1500,    # 前 1500 步作为缓冲期
    warmup_ratio=1e-6     # 从极其微小的学习率开始，慢慢爬升到设定的初始学习率
)

# 跑多少次迭代
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=3)
evaluation = dict(interval=1000, metric='mIoU')

# 官方基础配置 (日志格式、分布式训练钩子等，这些没法手写，可以直接借用默认的)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/zengyuzhi/project/nutrition/work_dirs/segment/iter_7500.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True 