# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import DETECTORS, build_neck
from .two_stage import TwoStageDetector

@DETECTORS.register_module()
class UDAAdversarialDetector(TwoStageDetector):
    """Adversarial UDA Detector for non-coregistered domains.
    
    It uses a DomainDiscriminator to align features between RGB and HSI.
    """

    def __init__(self,
                 backbone,
                 domain_discriminator,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(UDAAdversarialDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.domain_discriminator = build_neck(domain_discriminator)

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
        In this detector, 'img' is RGB (Source) and 'hsi' is HSI (Target).
        They are NOT necessarily co-registered.
        """
        batch_size_rgb = img.size(0)
        
        # 1. Extract Features for RGB (Source)
        # RGB bypasses the projector and goes directly to the inner resnet
        if hasattr(self.backbone, 'resnet'):
            x_rgb = self.backbone.resnet(img)
        else:
            x_rgb = self.backbone(img)
            
        if self.with_neck:
            x_rgb = self.neck(x_rgb)
        
        # 2. Extract Features for HSI (Target)
        if hsi is not None:
            # HSI MUST pass through the projector
            if hasattr(self.backbone, 'channel_projector'):
                import torch.nn.functional as F
                hsi_projected = self.backbone.channel_projector(hsi)
                if hasattr(self.backbone, 'target_size') and self.backbone.target_size:
                    hsi_projected = F.interpolate(hsi_projected, size=self.backbone.target_size, mode='bilinear')
                
                if hasattr(self.backbone, 'resnet'):
                    x_hsi = self.backbone.resnet(hsi_projected)
                else:
                    x_hsi = self.backbone(hsi_projected)
            else:
                x_hsi = self.backbone(hsi)
            
            if self.with_neck:
                x_hsi = self.neck(x_hsi)
        else:
            x_hsi = None

        losses = dict()

        # 3. Supervised Detection Loss (ONLY RGB)
        # RPN
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_rgb, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ROI
        roi_losses = self.roi_head.forward_train(
            x_rgb, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        # 4. Adversarial Domain Loss
        # We classify features as RGB (0) or HSI (1)
        if x_hsi is not None:
            # Domain labels: RGB is 0, HSI is 1
            domain_label_rgb = torch.zeros(batch_size_rgb, 1, device=img.device)
            domain_label_hsi = torch.ones(hsi.size(0), 1, device=hsi.device)
            
            # Forward pass through discriminator
            # The Discriminator contains the GRL, so it handles the adversarial gradient reversal
            domain_score_rgb = self.domain_discriminator(x_rgb)
            domain_score_hsi = self.domain_discriminator(x_hsi)
            
            # Loss: Binary Cross Entropy
            import torch.nn.functional as F
            loss_domain_rgb = F.binary_cross_entropy_with_logits(domain_score_rgb, domain_label_rgb)
            loss_domain_hsi = F.binary_cross_entropy_with_logits(domain_score_hsi, domain_label_hsi)
            
            losses['loss_adversarial_domain'] = (loss_domain_rgb + loss_domain_hsi) / 2.0

        return losses

    def simple_test(self, img, img_metas, hsi=None, rescale=False):
        """Test function without test-time augmentation.
        
        In UDA evaluation, if HSI is provided, we evaluate the detector's 
        performance on the Target Domain (HSI).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        if hsi is not None:
            if isinstance(hsi, list):
                hsi = hsi[0]
            # Evaluate on HSI (Target Domain)
            if hasattr(self.backbone, 'channel_projector'):
                import torch.nn.functional as F
                x = self.backbone.channel_projector(hsi)
                if hasattr(self.backbone, 'target_size') and self.backbone.target_size:
                    x = F.interpolate(x, size=self.backbone.target_size, mode='bilinear')
                
                if hasattr(self.backbone, 'resnet'):
                    x = self.backbone.resnet(x)
                else:
                    x = self.backbone(x)
            else:
                x = self.backbone(hsi)
        else:
            # Evaluate on RGB (Source Domain)
            if isinstance(img, list):
                img = img[0]
            if hasattr(self.backbone, 'resnet'):
                x = self.backbone.resnet(img)
            else:
                x = self.backbone(img)

        if isinstance(img_metas, list) and isinstance(img_metas[0], list):
             img_metas = img_metas[0]

        if self.with_neck:
            x = self.neck(x)

        proposal_list = self.rpn_head.simple_test(x, img_metas, rescale=rescale)
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test-time augmentation."""
        raise NotImplementedError
