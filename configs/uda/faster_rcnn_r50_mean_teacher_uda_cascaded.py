# Mean Teacher UDA with Cascaded Spectral Projector
# Ablation: replaces 1x1 bottleneck (30→3) with cascaded (30→16→3 + attention)
_base_ = [
    '../_base_/datasets/coco_uda_hsi.py',
    '../_base_/default_runtime.py'
]

load_from = 'work_dirs/faster_rcnn_r50_rgb/rgb_projected.pth'

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='MeanTeacherUDADetector',
    # --- Mean Teacher Params (same as baseline) ---
    ema_decay=0.9996,
    pseudo_thr=0.9,
    pseudo_thr_min=0.7,
    pseudo_warmup_epochs=10,
    burn_in_epochs=5,
    pseudo_loss_weight=1.0,
    domain_loss_weight=0.1,
    spectral_dropout_rate=0.3,
    # --- Backbone: CASCADED ProjectUpsampleResNet ---
    backbone=dict(
        type='CascadedProjectUpsampleResNet',
        in_channels=30,
        mid_channels=16,          # Intermediate spectral channels
        attention_reduction=4,    # SE reduction ratio
        target_size=(1600, 1600),
        resnet_cfg=dict(
            type='ResNet',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            zero_init_residual=False,
            with_cp=True,
            norm_eval=True,
            style='pytorch'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        with_cp=False,
        norm_cfg=norm_cfg
    ),
    instance_discriminator=dict(
        type='InstanceDomainDiscriminator',
        in_channels=256,
        roi_feat_size=7,
        hidden_dim=1024,
        grl_alpha=0.1
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=512, pos_fraction=0.25,
                neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.9), min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.01),
            max_per_img=100))
)

data = dict(samples_per_gpu=1, workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500,
    warmup_ratio=0.001, step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=1)

custom_hooks = [
    dict(type='MeanTeacherEpochHook', priority='NORMAL')
]
