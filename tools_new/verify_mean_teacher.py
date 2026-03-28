"""Verify the Mean Teacher UDA system end-to-end with mock data.

Tests:
  1. Training forward pass (all losses computed)
  2. EMA teacher update
  3. Pseudo-label generation
  4. Evaluation (simple_test) on HSI
"""
import sys
import torch
sys.path.append('.')

from mmcv.utils import ConfigDict
from mmdet.models import build_detector


def build_mock_model():
    """Build a small model with mock config for fast testing."""
    norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    cfg = ConfigDict(dict(
        type='MeanTeacherUDADetector',
        ema_decay=0.999,
        pseudo_thr=0.5,       # Low for testing
        pseudo_thr_min=0.3,
        pseudo_warmup_epochs=2,
        burn_in_epochs=0,     # No burn-in for testing
        pseudo_loss_weight=1.0,
        domain_loss_weight=0.1,
        spectral_dropout_rate=0.3,
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
        instance_discriminator=dict(
            type='InstanceDomainDiscriminator',
            in_channels=256,
            roi_feat_size=7,
            hidden_dim=256,
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
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7,
                              neg_iou_thr=0.3, min_pos_iou=0.3,
                              match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=256,
                             pos_fraction=0.5, add_gt_as_proposals=False),
                allowed_border=-1, pos_weight=-1, debug=False),
            rpn_proposal=dict(nms_pre=200, max_per_img=100,
                              nms=dict(type='nms', iou_threshold=0.7),
                              min_bbox_size=0),
            rcnn=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5,
                              neg_iou_thr=0.5, min_pos_iou=0.5),
                sampler=dict(type='RandomSampler', num=128,
                             pos_fraction=0.25, add_gt_as_proposals=False),
                pos_weight=-1, debug=False)),
        test_cfg=dict(
            rpn=dict(nms_pre=200, max_per_img=100,
                     nms=dict(type='nms', iou_threshold=0.9),
                     min_bbox_size=0),
            rcnn=dict(score_thr=0.01,
                      nms=dict(type='soft_nms', iou_threshold=0.5),
                      max_per_img=100))
    ))
    return build_detector(cfg)


def make_mock_data(device='cpu'):
    """Create mock RGB + HSI data."""
    img = torch.randn(1, 3, 400, 400, device=device)
    hsi = torch.randn(1, 30, 224, 224, device=device)
    img_metas = [dict(
        img_shape=(400, 400, 3),
        ori_shape=(400, 400, 3),
        pad_shape=(400, 400, 3),
        scale_factor=1.0
    )]
    gt_bboxes = [torch.tensor([[50, 50, 200, 200]], dtype=torch.float32,
                               device=device)]
    gt_labels = [torch.tensor([0], dtype=torch.long, device=device)]
    return img, hsi, img_metas, gt_bboxes, gt_labels


def test_training(model, device='cpu'):
    """Test Phase 1: Training forward pass."""
    print('\n=== Phase 1: Training Forward ===')
    model.train()
    img, hsi, img_metas, gt_bboxes, gt_labels = make_mock_data(device)

    losses = model.forward_train(img, img_metas, gt_bboxes, gt_labels, hsi=hsi)
    print('Losses computed:')
    for k, v in sorted(losses.items()):
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.item():.4f}')
        elif isinstance(v, list):
            print(f'  {k}: {sum(x.item() for x in v):.4f}')
    
    # Verify expected losses are present
    assert 'loss_rpn_cls' in losses, 'Missing loss_rpn_cls'
    assert 'loss_cls' in losses, 'Missing loss_cls'
    print('✓ Supervised losses present')
    
    # Domain loss should be present if discriminator is enabled
    if model.instance_discriminator is not None:
        assert 'loss_domain' in losses, 'Missing loss_domain'
        print('✓ Adversarial domain loss present')
    
    print('✓ Phase 1 PASSED')
    return losses


def test_ema_update(model):
    """Test Phase 2: EMA teacher update."""
    print('\n=== Phase 2: EMA Update ===')
    # Get teacher param snapshot before update
    t_params_before = [p.clone() for p in model.teacher_backbone.parameters()]
    
    # Modify student params slightly
    for p in model.backbone.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    
    # Update teacher
    model.update_teacher()
    
    # Verify teacher was updated (not identical to before)
    changed = False
    for t_before, t_after in zip(t_params_before,
                                  model.teacher_backbone.parameters()):
        if not torch.equal(t_before, t_after.data):
            changed = True
            break
    assert changed, 'Teacher parameters did not change after EMA update'
    print('✓ Teacher parameters updated via EMA')
    print('✓ Phase 2 PASSED')


def test_pseudo_labels(model, device='cpu'):
    """Test Phase 3: Pseudo-label generation."""
    print('\n=== Phase 3: Pseudo-Label Generation ===')
    model.eval()
    hsi = torch.randn(1, 30, 224, 224, device=device)
    img_metas = [dict(
        img_shape=(400, 400, 3),
        ori_shape=(400, 400, 3),
        pad_shape=(400, 400, 3),
        scale_factor=1.0
    )]
    
    with torch.no_grad():
        pseudo_bboxes, pseudo_labels = model.generate_pseudo_labels(
            hsi, img_metas)
    
    assert len(pseudo_bboxes) == 1, f'Expected 1 image, got {len(pseudo_bboxes)}'
    assert len(pseudo_labels) == 1, f'Expected 1 image, got {len(pseudo_labels)}'
    
    n = len(pseudo_bboxes[0])
    print(f'  Generated {n} pseudo-labels (may be 0 with random weights)')
    if n > 0:
        print(f'  Bbox shape: {pseudo_bboxes[0].shape}')
        print(f'  Labels: {pseudo_labels[0]}')
    print('✓ Phase 3 PASSED')


def test_evaluation(model, device='cpu'):
    """Test Phase 4: Evaluation (simple_test)."""
    print('\n=== Phase 4: Evaluation ===')
    model.eval()
    img = torch.randn(1, 3, 400, 400, device=device)
    hsi = torch.randn(1, 30, 224, 224, device=device)
    img_metas = [dict(
        img_shape=(400, 400, 3),
        ori_shape=(400, 400, 3),
        pad_shape=(400, 400, 3),
        scale_factor=1.0
    )]
    
    with torch.no_grad():
        # Test with list-wrapped inputs (as mmdet evaluation provides)
        results = model.simple_test([img], [img_metas], hsi=[hsi])
    
    assert len(results) == 1, f'Expected 1 result, got {len(results)}'
    assert len(results[0]) == 2, f'Expected 2 classes, got {len(results[0])}'
    print(f'  Detection results: {len(results[0])} categories')
    print('✓ Phase 4 PASSED')


def main():
    print('=' * 60)
    print('Mean Teacher UDA Detector: End-to-End Verification')
    print('=' * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print('\nBuilding model...')
    model = build_mock_model()
    model = model.to(device)
    print(f'Model built. Student params: '
          f'{sum(p.numel() for p in model.parameters()):,}')
    print(f'Teacher params: '
          f'{sum(p.numel() for p in model._teacher_parameters()):,}')

    test_training(model, device)
    test_ema_update(model)
    test_pseudo_labels(model, device)
    test_evaluation(model, device)

    print('\n' + '=' * 60)
    print('ALL PHASES PASSED ✓')
    print('=' * 60)


if __name__ == '__main__':
    main()
