import torch
import sys
import os
sys.path.append('.')

from mmdet.models.detectors.joint_modality_detector import JointModalityDetector
from mmcv.utils import Config, ConfigDict

def test_joint_forward():
    print("Testing JointModalityDetector forward pass...")
    
    # Mock configuration
    norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    model_cfg = dict(
        type='JointModalityDetector',
        consistency_weight=0.1,
        backbone=dict(
            type='ProjectUpsampleResNet',
            in_channels=30,
            target_size=(400, 400),
            resnet_cfg=dict(
                type='ResNet',
                in_channels=3,
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=norm_cfg,
                style='pytorch'
            )
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            norm_cfg=norm_cfg
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
            bbox_coder=dict(type='DeltaXYWHBBoxCoder'),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True),
            loss_bbox=dict(type='SmoothL1Loss')
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            bbox_head=dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                num_classes=2,
                norm_cfg=norm_cfg
            )
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.9)),
            rcnn=dict(score_thr=0.01, nms=dict(type='soft_nms', iou_threshold=0.5), max_per_img=100)
        )
    )

    from mmdet.models import build_detector
    model_cfg = ConfigDict(model_cfg)
    detector = build_detector(model_cfg)
    detector.train()

    # Create dummy inputs (reduced size to avoid OOM)
    img = torch.randn(2, 3, 400, 400)
    hsi = torch.randn(2, 30, 224, 224)
    img_metas = [
        dict(img_shape=(400, 400, 3), ori_shape=(400, 400, 3), pad_shape=(400, 400, 3), scale_factor=1.0),
        dict(img_shape=(400, 400, 3), ori_shape=(400, 400, 3), pad_shape=(400, 400, 3), scale_factor=1.0)
    ]
    gt_bboxes = [torch.tensor([[10, 10, 100, 100]], dtype=torch.float32) for _ in range(2)]
    gt_labels = [torch.tensor([0], dtype=torch.long) for _ in range(2)]

    print("Running forward_train...")
    losses = detector.forward_train(img, img_metas, gt_bboxes, gt_labels, hsi=hsi)

    print("\nLosses computed:")
    for k, v in losses.items():
        if isinstance(v, list):
            print(f"{k}: {[val.item() for val in v]}")
        else:
            print(f"{k}: {v.item()}")

    assert 'loss_consistency' in losses, "Consistency loss missing!"
    print("\nTest passed successfully!")

if __name__ == '__main__':
    test_joint_forward()
