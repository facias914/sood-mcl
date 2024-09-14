angle_version = 'le90'

import torchvision.transforms as transforms
from copy import deepcopy

# model settings
detector = dict(
    type='SemiRotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='SemiRotatedFCOSHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

model = dict(
    type="RotatedSSTGDenseTeacher",
    model=detector,
    semi_loss=dict(type='RotatedSingleStageDTLoss', loss_type='pr_origin_p5',
                   cls_loss_type='bce', dynamic_weight='50ang',
                   aux_loss='ot_loss_norm', aux_loss_cfg=dict(clamp_ot=True)),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=16000,
        sup_weight=1.0,
        unsup_weight=1.0,
        weight_suppress="linear",
        logit_specific_weights=dict(),
        region_ratio=0.01
    ),
    test_cfg=dict(inference_on="teacher"),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
common_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
strong_pipeline = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply', operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply', operations=[
        dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])
    ]),
    # dict(type='DTRandCrop'),
    dict(type='DTToNumpy'),
    dict(type="ExtraAttrs", tag="unsup_strong"),
]
weak_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="unsup_weak"),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    dict(type="LoadEmptyAnnotations", with_bbox=True),
    dict(type="STMultiBranch", unsup_strong=deepcopy(strong_pipeline), unsup_weak=deepcopy(weak_pipeline),
         common_pipeline=common_pipeline, is_seq=True),  
]
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type="ExtraAttrs", tag="sup_weak"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg', 'tag')
         )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'DOTADataset'   
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=5,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type=dataset_type,
            ann_file="/workspace/DOTA/v15/train_split/labelTxt/",
            img_prefix="/workspace/DOTA/v15/train_split/images/",
            classes=classes,
            pipeline=sup_pipeline,
        ),
        unsup=dict(
            type=dataset_type,
            ann_file="/workspace/DOTA/v15/test_split/empty_annfiles/",
            img_prefix="/workspace/DOTA/v15/test_split/images/",
            classes=classes,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        img_prefix="/workspace/DOTA/v15/val_split/images/",
        ann_file='/workspace/DOTA/v15/val_split/labelTxt/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        img_prefix="/workspace/DOTA/v15/val_split/images/",
        ann_file='/workspace/DOTA/v15/val_split/labelTxt/',
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(
        train=dict(
            type="GroupMultiSourceSampler",
            sample_ratio=[2, 1]
        )
    ),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9996, interval=1, start_steps=3200),
]

# evaluation
evaluation = dict(type="SubModulesDistEvalHook", interval=3200, metric='mAP',
                  save_best='mAP')

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[120000, 160000])
# 120k iters is enough for DOTA
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=3200, max_keep_ckpts=5)

# Default: disable fp16 training
# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# resume_from = None
workflow = [('train', 1)]   # mode, iters

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'