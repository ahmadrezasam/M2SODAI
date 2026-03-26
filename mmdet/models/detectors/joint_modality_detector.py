# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class JointModalityDetector(TwoStageDetector):
    """Joint Modality Detector for Domain Adaptation.
    
    It supports co-registered RGB and HSI inputs. During training, it performs
    a dual-forward pass to align HSI features with RGB features.
    """

    def __init__(self,
                backbone,
                neck=None,
                rpn_head=None,
                roi_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                init_cfg=None,
                consistency_weight=0.1):
        super(JointModalityDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.consistency_weight = consistency_weight

    def extract_feat(self, img, modality='rgb'):
        """Extract features from the backbone+neck, handling modalities."""
        if modality == 'hsi':
            # HSI passes through projector + resnet inside ProjectUpsampleResNet
            x = self.backbone(img)
        else:
            # RGB bypasses projector and passes directly to the inner resnet
            if hasattr(self.backbone, 'resnet'):
                x = self.backbone.resnet(img)
            else:
                x = self.backbone(img)
        
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                     img,
                     img_metas,
                     gt_bboxes,
                     gt_labels,
                     hsi=None,
                     gt_bboxes_ignore=None,
                     gt_masks=None,
                     proposals=None,
                     **kwargs):
        """
        Args:
            img (Tensor): RGB images (N, 3, H, W).
            hsi (Tensor, optional): HSI cubes (N, C_hsi, H_hsi, W_hsi).
        """
        batch_size = img.shape[0]
        
        # 1. Prepare inputs for concatenated forward pass
        if hsi is not None:
            # HSI passes through projector + upsampling
            hsi_projected = self.backbone.channel_projector(hsi)
            if self.backbone.target_size and hsi_projected.shape[2:] != tuple(self.backbone.target_size):
                hsi_projected = F.interpolate(hsi_projected, size=self.backbone.target_size, 
                                            mode='bilinear', align_corners=False)
            joint_img = torch.cat([img, hsi_projected], dim=0)
        else:
            joint_img = img

        # 2. Extract features once for the joint batch
        if hasattr(self.backbone, 'resnet'):
            x_joint = self.backbone.resnet(joint_img)
        else:
            x_joint = self.backbone(joint_img)
            
        if self.with_neck:
            x_joint = self.neck(x_joint)
        
        # 3. Split features back into RGB and HSI
        if hsi is not None:
            x_rgb = [f[:batch_size] for f in x_joint]
            x_hsi = [f[batch_size:] for f in x_joint]
        else:
            x_rgb = x_joint
            x_hsi = None

        losses = dict()
        
        # 4. Concatenate targets for Heads if HSI is present
        if hsi is not None:
            # We use co-reg labels for both modalities
            gt_bboxes_head = gt_bboxes + gt_bboxes
            gt_labels_head = gt_labels + gt_labels
            img_metas_head = img_metas + img_metas
            # Handle gt_bboxes_ignore and gt_masks if they exist
            gt_bboxes_ignore_head = gt_bboxes_ignore + gt_bboxes_ignore if gt_bboxes_ignore is not None else None
            gt_masks_head = gt_masks + gt_masks if gt_masks is not None else None
        else:
            gt_bboxes_head = gt_bboxes
            gt_labels_head = gt_labels
            img_metas_head = img_metas
            gt_bboxes_ignore_head = gt_bboxes_ignore
            gt_masks_head = gt_masks

        # 5. RPN Forward and Loss (Joint)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_joint, img_metas_head, gt_bboxes_head, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore_head, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # 6. ROI Forward and Loss (Joint)
        roi_losses = self.roi_head.forward_train(
            x_joint, img_metas_head, proposal_list, gt_bboxes_head, gt_labels_head,
            gt_bboxes_ignore_head, gt_masks_head, **kwargs)
        losses.update(roi_losses)

        # 7. Cross-Modality Consistency Loss
        if hsi is not None and self.consistency_weight > 0:
            consistency_loss = 0
            # x_rgb and x_hsi are slices of x_joint features
            for feat_rgb, feat_hsi in zip(x_rgb, x_hsi):
                consistency_loss += F.mse_loss(feat_hsi, feat_rgb.detach())
            losses['loss_consistency'] = consistency_loss * self.consistency_weight

        return losses

    def simple_test(self, img, img_metas, hsi=None, proposals=None, rescale=False):
        """Test with either RGB or HSI."""
        if hsi is not None:
            # Handle list-wrapped HSI from data loader if necessary
            if isinstance(hsi, list):
                hsi = hsi[0]
            # If both are provided, we prefer HSI if we are testing HSI performance
            # or we could do ensemble. Here we follow the modality provided.
            x = self.extract_feat(hsi, modality='hsi')
        else:
            x = self.extract_feat(img, modality='rgb')
            
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
