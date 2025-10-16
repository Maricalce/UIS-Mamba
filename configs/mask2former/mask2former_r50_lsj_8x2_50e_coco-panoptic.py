# _base_ = [
#     '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
# ]
# num_things_classes = 7
# num_stuff_classes = 0
# # num_things_classes = 80
# # num_stuff_classes = 53
# num_classes = num_things_classes + num_stuff_classes
# model = dict(
#     type='MM_GrootV',
#     backbone=dict(
#         type='MM_GrootV',  # 修改为注册的类名
#         out_indices=(0, 1, 2, 3),
#         pretrained="/root/data1/yzj/WaterMask/tools/grootv_cls_tiny.pth",  # 更新预训练路径
#         # GrootV特有参数
#         channels=80,  # 原dims参数改名
#         depths=[2, 2, 9, 2],  # 保持深度配置
#         mlp_ratio=4.0,
#         drop_rate=0.0,  # 新增参数
#         drop_path_rate=0.1,
#         act_layer='GELU',  # 明确激活函数类型
#         norm_layer='LN',  # 统一使用LayerNorm
#         post_norm=False,  # 新增后标准化配置
#         with_cp=False,  # 检查点配置
#         # Tree-SSM相关参数
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2,
#         d_conv=3,
#         conv_bias=False,
#         dt_rank="auto"
#     ),
#     panoptic_head=dict(
#         type='Mask2FormerHead',
#         in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
#         strides=[4, 8, 16, 32],
#         feat_channels=256,
#         out_channels=256,
#         num_things_classes=num_things_classes,
#         num_stuff_classes=num_stuff_classes,
#         num_queries=100,
#         num_transformer_feat_level=3,
#         pixel_decoder=dict(
#             type='MSDeformAttnPixelDecoder',
#             num_outs=3,
#             norm_cfg=dict(type='GN', num_groups=32),
#             act_cfg=dict(type='ReLU'),
#             encoder=dict(
#                 type='DetrTransformerEncoder',
#                 num_layers=6,
#                 transformerlayers=dict(
#                     type='BaseTransformerLayer',
#                     attn_cfgs=dict(
#                         type='MultiScaleDeformableAttention',
#                         embed_dims=256,
#                         num_heads=8,
#                         num_levels=3,
#                         num_points=4,
#                         im2col_step=64,
#                         dropout=0.0,
#                         batch_first=False,
#                         norm_cfg=None,
#                         init_cfg=None),
#                     ffn_cfgs=dict(
#                         type='FFN',
#                         embed_dims=256,
#                         feedforward_channels=1024,
#                         num_fcs=2,
#                         ffn_drop=0.0,
#                         act_cfg=dict(type='ReLU', inplace=True)),
#                     operation_order=('self_attn', 'norm', 'ffn', 'norm')),
#                 init_cfg=None),
#             positional_encoding=dict(
#                 type='SinePositionalEncoding', num_feats=128, normalize=True),
#             init_cfg=None),
#         enforce_decoder_input_project=False,
#         positional_encoding=dict(
#             type='SinePositionalEncoding', num_feats=128, normalize=True),
#         transformer_decoder=dict(
#             type='DetrTransformerDecoder',
#             return_intermediate=True,
#             num_layers=9,
#             transformerlayers=dict(
#                 type='DetrTransformerDecoderLayer',
#                 attn_cfgs=dict(
#                     type='MultiheadAttention',
#                     embed_dims=256,
#                     num_heads=8,
#                     attn_drop=0.0,
#                     proj_drop=0.0,
#                     dropout_layer=None,
#                     batch_first=False),
#                 ffn_cfgs=dict(
#                     embed_dims=256,
#                     feedforward_channels=2048,
#                     num_fcs=2,
#                     act_cfg=dict(type='ReLU', inplace=True),
#                     ffn_drop=0.0,
#                     dropout_layer=None,
#                     add_identity=True),
#                 feedforward_channels=2048,
#                 operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
#                                  'ffn', 'norm')),
#             init_cfg=None),
#         loss_cls=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=2.0,
#             reduction='mean',
#             class_weight=[1.0] * num_classes + [0.1]),
#         loss_mask=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='mean',
#             loss_weight=5.0),
#         loss_dice=dict(
#             type='DiceLoss',
#             use_sigmoid=True,
#             activate=True,
#             reduction='mean',
#             naive_dice=True,
#             eps=1.0,
#             loss_weight=5.0)),
#     panoptic_fusion_head=dict(
#         type='MaskFormerFusionHead',
#         num_things_classes=num_things_classes,
#         num_stuff_classes=num_stuff_classes,
#         loss_panoptic=None,
#         init_cfg=None),
#     train_cfg=dict(
#         num_points=12544,
#         oversample_ratio=3.0,
#         importance_sample_ratio=0.75,
#         assigner=dict(
#             type='MaskHungarianAssigner',
#             cls_cost=dict(type='ClassificationCost', weight=2.0),
#             mask_cost=dict(
#                 type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
#             dice_cost=dict(
#                 type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
#         sampler=dict(type='MaskPseudoSampler')),
#     test_cfg=dict(
#         panoptic_on=True,
#         # For now, the dataset does not support
#         # evaluating semantic segmentation metric.
#         semantic_on=False,
#         instance_on=True,
#         # max_per_image is for instance segmentation.
#         max_per_image=100,
#         iou_thr=0.8,
#         # In Mask2Former's panoptic postprocessing,
#         # it will filter mask area where score is less than 0.5 .
#         filter_low_score=True),
#     init_cfg=None)

# # dataset settings
# image_size = (1024, 1024)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(
#         type='LoadPanopticAnnotations',
#         with_bbox=True,
#         with_mask=True,
#         with_seg=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     # large scale jittering
#     dict(
#         type='Resize',
#         img_scale=image_size,
#         ratio_range=(0.1, 2.0),
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_size=image_size,
#         crop_type='absolute',
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=image_size),
#     dict(type='DefaultFormatBundle', img_to_float=True),
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data_root = 'data/coco/'
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=0,
#     train=dict(pipeline=train_pipeline),
#     val=dict(
#         pipeline=test_pipeline,
#         ins_ann_file=data_root + 'annotations/instances_val2017.json',
#     ),
#     test=dict(
#         pipeline=test_pipeline,
#         ins_ann_file=data_root + 'annotations/instances_val2017.json',
#     ))

# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# # optimizer
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     weight_decay=0.05,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#         },
#         norm_decay_mult=0.0))
# optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# # learning policy
# lr_config = dict(
#     policy='step',
#     gamma=0.1,
#     by_epoch=False,
#     step=[327778, 355092],
#     warmup='linear',
#     warmup_by_epoch=False,
#     warmup_ratio=1.0,  # no warmup
#     warmup_iters=10)

# max_iters = 368750
# # runner = dict(type='IterBasedRunner', max_iters=max_iters)
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='TensorboardLoggerHook', by_epoch=False)
#     ])
# interval = 5000
# workflow = [('train', interval)]
# checkpoint_config = dict(
#     by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=3)

# # Before 365001th iteration, we do evaluation every 5000 iterations.
# # After 365000th iteration, we do evaluation every 368750 iterations,
# # which means that we do evaluation at the end of training.
# dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
# evaluation = dict(
#     interval=interval,
#     dynamic_intervals=dynamic_intervals,
#     metric=['PQ', 'bbox', 'segm'])
_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
num_things_classes = 7
num_stuff_classes = 0
# num_things_classes = 80
# num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2Former',  # 修改为 Mask2Former，而不是 MM_GrootV
    backbone=dict(
        type='MM_GrootV',  # 修改为正确的骨干网络类型
        out_indices=(0, 1, 2, 3),
        pretrained="/root/data1/yzj/WaterMask/tools/grootv_cls_tiny.pth",
        # GrootV特有参数
        channels=80,
        depths=[2, 2, 9, 2],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer='GELU',
        norm_layer='LN',
        post_norm=False,
        with_cp=False,
        # Tree-SSM相关参数
        ssm_ratio=2.0,
        ssm_rank_ratio=2,
        d_conv=3,
        conv_bias=False,
        dt_rank="auto"
    ),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[80, 160, 320, 640],  # 根据GrootV输出通道调整
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = '/root/data1/yzj/USIS10K/'  # 更新数据根路径
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "multi_class_annotations/multi_class_train_annotations.json",
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        ins_ann_file=data_root + 'annotations/instances_val2017.json',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        ins_ann_file=data_root + 'annotations/instances_val2017.json',
    ),
    # 显式设置数据加载器配置
    train_dataloader=dict(
        batch_size=2,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    ),
    val_dataloader=dict(
        batch_size=2,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    ),
    test_dataloader=dict(
        batch_size=2,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )
)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy - 基于轮次的学习率策略
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=True,  # 修改为基于轮次
    step=[24, 32],  # 根据总轮次调整
    warmup='linear',
    warmup_by_epoch=True,  # 修改为基于轮次
    warmup_ratio=1e-3,
    warmup_iters=1)  # 1个轮次的预热

runner = dict(type='EpochBasedRunner', max_epochs=36)  # 明确使用EpochBasedRunner

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),  # 修改为基于轮次
        dict(type='TensorboardLoggerHook', by_epoch=True)  # 修改为基于轮次
    ])

# 基于轮次的评估和保存间隔
interval = 1  # 每1个轮次评估一次
workflow = [('train', 1)]  # 每个轮次训练一次
checkpoint_config = dict(
    by_epoch=True, interval=1, save_last=True, max_keep_ckpts=3)

# 评估配置
evaluation = dict(
    interval=1,  # 每1个轮次评估一次
    metric=['PQ', 'bbox', 'segm'])