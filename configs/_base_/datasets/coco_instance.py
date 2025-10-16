# # dataset settings
# dataset_type = 'CocoDataset'
# # data_root = '/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/UIIS/UDW/'
# data_root ="/root/data1/yzj/UIIS/UDW/"
# # data_root ="/root/data1/yzj/USIS10K/"
# # data_root ="/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/USIS10K/"
# img_norm_cfg = dict(
#     mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/train.json',
#         # ann_file=data_root + "multi_class_annotations/multi_class_train_annotations.json",
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         # ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         # ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])

# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/UIIS/UDW/'
# data_root ="/root/data1/yzj/UIIS/UDW/"
data_root ="/root/data1/yzj/USIS10K/"
# data_root ="/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/USIS10K/"
img_norm_cfg = dict(
    mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/train.json',
        # ann_file=data_root + "foreground_annotations/foreground_train_annotations.json",
        ann_file=data_root + "multi_class_annotations/multi_class_train_annotations.json",
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        # ann_file=data_root + "foreground_annotations/foreground_val_annotations.json",
        ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        # ann_file=data_root + "foreground_annotations/foreground_val_annotations.json",
        ann_file=data_root + "multi_class_annotations/multi_class_val_annotations.json",
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])


# # dataset settings
# dataset_type = 'CocoDataset'
# # data_root = '/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/UIIS/UDW/'
# # data_root ="/root/data1/yzj/UIIS/UDW/"
# data_root ="/root/data1/yzj/coco/"
# # data_root ="/opt/data/private/mvp_undergraduate2024/YuZongji/mamba/VMamba-main/data/USIS10K/"
# img_norm_cfg = dict(
#     mean = [81.236, 113.761, 117.095], std = [60.598, 58.471, 62.821], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=0,
#     train=dict(
#         type=dataset_type,
#         # ann_file=data_root + 'annotations/train.json',
#         # ann_file=data_root + "foreground_annotations/foreground_train_annotations.json",
#         ann_file=data_root + "annotations/instances_train2017.json",
#         img_prefix=data_root + 'train2017/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         # ann_file=data_root + 'annotations/val.json',
#         # ann_file=data_root + "foreground_annotations/foreground_val_annotations.json",
#         ann_file=data_root + "annotations/instances_val2017.json",
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         # ann_file=data_root + 'annotations/val.json',
#         # ann_file=data_root + "foreground_annotations/foreground_val_annotations.json",
#         ann_file=data_root + "annotations/instances_val2017.json",
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])
