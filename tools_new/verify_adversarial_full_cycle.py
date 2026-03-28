import torch
import sys
import os
sys.path.append('.')

from mmdet.models.detectors.uda_adversarial_detector import UDAAdversarialDetector
from mmcv.utils import Config, ConfigDict

def verify_full_cycle():
    print("Starting full-cycle verification (Train + Eval)...")
    
    # Mock configuration
    norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    model_cfg = dict(
        type='UDAAdversarialDetector',
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
        domain_discriminator=dict(
            type='DomainDiscriminator',
            in_channels=256,
            hidden_channels=128,
            num_convs=2,
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
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, add_gt_as_proposals=False),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.9), min_bbox_size=0),
            rcnn=dict(score_thr=0.01, nms=dict(type='soft_nms', iou_threshold=0.5), max_per_img=100)
        )
    )

    from mmdet.models import build_detector
    model_cfg = ConfigDict(model_cfg)
    detector = build_detector(model_cfg)

    # 1. TEST TRAINING STEP
    print("\n--- Phase 1: Training Step ---")
    detector.train()
    batch_size = 1
    img = torch.randn(batch_size, 3, 400, 400)
    hsi = torch.randn(batch_size, 30, 224, 224)
    img_metas = [
        dict(img_shape=(400, 400, 3), ori_shape=(400, 400, 3), pad_shape=(400, 400, 3), scale_factor=1.0)
    ]
    gt_bboxes = [torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    losses = detector.forward_train(img, img_metas, gt_bboxes, gt_labels, hsi=hsi)
    print("Training losses computed successfully:")
    for k, v in losses.items():
        print(f"  {k}: {round(v.item() if isinstance(v, torch.Tensor) else sum(v).item(), 4)}")

    # 2. TEST EVALUATION (SIMPLE_TEST)
    print("\n--- Phase 2: Evaluation Step (Simple Test) ---")
    detector.eval()
    
    # Simulate list-wrapped inputs as provided by MMDetection Evaluation Hook
    img_list = [img]
    hsi_list = [hsi]
    img_metas_list = [img_metas]
    
    print("Calling simple_test with list-wrapped HSI...")
    try:
        with torch.no_grad():
            results = detector.simple_test(img_list, img_metas_list, hsi=hsi_list)
        print("Evaluation results computed successfully!")
        print(f"Detection results for first image: {len(results[0])} categories found.")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nVerification PASSED! Both training and evaluation paths are working correctly.")

if __name__ == '__main__':
    verify_full_cycle()
