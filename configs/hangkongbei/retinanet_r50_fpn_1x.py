# model settings
model = dict(
    type='RetinaNet',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.3),
    max_per_img=100)
# dataset settings
# dataset settings
dataset_type = 'voc_hangkongbeiDataset'
data_root = '../dataset/hangkongbei/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',  # to avoid reloading datasets frequently
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                #data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/'],
            img_scale=(800, 800),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True,
            #data aug
            # extra_aug=dict(
            #     photo_metric_distortion=dict(
            #         brightness_delta=32,
            #         contrast_range=(0.8, 1.2),
            #         saturation_range=(0.8, 1.2),
            #         hue_delta=18),
            #         # expand=dict(
            #         #     mean=img_norm_cfg['mean'],
            #         #     to_rgb=img_norm_cfg['to_rgb'],
            #         #     ratio_range=(1, 4)),
            #         # random_crop=dict(
            #         #     min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.6)
            #     ),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
       ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False
       ))

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# data = dict(
#     imgs_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_train2017.json',
#         img_prefix=data_root + 'train2017/',
#         img_scale=(1333, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0.5,
#         with_mask=False,
#         with_crowd=False,
#         with_label=True),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         img_scale=(1333, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0,
#         with_mask=False,
#         with_crowd=False,
#         with_label=True),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         img_scale=(1333, 800),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0,
#         with_mask=False,
#         with_crowd=False,
#         with_label=False,
#         test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(0)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x_800_square'
load_from = None
resume_from = None
workflow = [('train', 1)]
