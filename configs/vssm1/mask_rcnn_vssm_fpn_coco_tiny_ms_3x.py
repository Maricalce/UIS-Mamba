_base_ = [
    # '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    '../_our_/water_r50_fpn_ms3x.py'
]

model = dict(
    backbone=dict(
        type='MM_GrootV',  # 修改为注册的类名
        out_indices=(0, 1, 2, 3),
        pretrained="/root/data1/yzj/WaterMask/tools/grootv_cls_tiny.pth",  # 更新预训练路径
        # GrootV特有参数
        channels=80,  # 原dims参数改名
        depths=[2, 2, 9, 2],  # 保持深度配置
        mlp_ratio=4.0,
        drop_rate=0.0,  # 新增参数
        drop_path_rate=0.1,
        act_layer='GELU',  # 明确激活函数类型
        norm_layer='LN',  # 统一使用LayerNorm
        post_norm=False,  # 新增后标准化配置
        with_cp=False,  # 检查点配置
        # Tree-SSM相关参数
        ssm_ratio=2.0,
        ssm_rank_ratio=2,
        d_conv=3,
        conv_bias=False,
        dt_rank="auto"
    ),
    neck=dict(
        in_channels=[80, 160, 320, 640]  # 与dims对应
        #in_channels=[96, 192, 384, 768]  # 与dims对应
    ),
    roi_head=dict(
        mask_head=dict(
            num_heads_in_gat=2
        )
    )

)
evaluation = dict(metric=['bbox','segm'], classwise=True, interval=1)
#
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
# lr_config = dict(warmup_iters=1000, step=[8, 11])
# runner = dict(max_epochs=36)
#
# # train_dataloader = dict(batch_size=2) # as gpus=8
#
# # augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[[
#             dict(
#                 type='RandomChoiceResize',
#                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                         (736, 1333), (768, 1333), (800, 1333)],
#                 keep_ratio=True)
#         ],
#                     [
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(400, 1333), (500, 1333), (600, 1333)],
#                             keep_ratio=True),
#                         dict(
#                             type='RandomCrop',
#                             crop_type='absolute_range',
#                             crop_size=(384, 600),
#                             allow_negative_crop=True),
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(480, 1333), (512, 1333), (544, 1333),
#                                     (576, 1333), (608, 1333), (640, 1333),
#                                     (672, 1333), (704, 1333), (736, 1333),
#                                     (768, 1333), (800, 1333)],
#                             keep_ratio=True)
#                     ]]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
#
# max_epochs = 48
# train_cfg = dict(max_epochs=max_epochs)
#
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[27, 33],
#         gamma=0.1)
# ]
#
# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0001,
#         betas=(0.9, 0.999),
#         weight_decay=0.05))

