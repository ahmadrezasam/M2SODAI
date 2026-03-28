# Copyright (c) OpenMMLab. All rights reserved.
import torch
from .joint_modality_detector import JointModalityDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class UDAJointModalityDetector(JointModalityDetector):
    """Unsupervised Joint Modality Detector.
    
    It inherits from JointModalityDetector but ensures that only RGB images
    are used for supervised detection losses (RPN and ROI). HSI images
    are used only for the cross-modality consistency loss.
    """

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
        
        # 1. Extract joint features (same as parent)
        # We reuse the parent's logic to handle projection and shared backbone
        if hsi is not None:
            import torch.nn.functional as F
            # HSI passes through projector + upsampling
            hsi_projected = self.backbone.channel_projector(hsi)
            if hasattr(self.backbone, 'target_size') and self.backbone.target_size and hsi_projected.shape[2:] != tuple(self.backbone.target_size):
                hsi_projected = F.interpolate(hsi_projected, size=self.backbone.target_size, 
                                            mode='bilinear', align_corners=False)
            joint_img = torch.cat([img, hsi_projected], dim=0)
        else:
            joint_img = img

        # Extract features for the joint batch
        if hasattr(self.backbone, 'resnet'):
            x_joint = self.backbone.resnet(joint_img)
        else:
            x_joint = self.backbone(joint_img)
            
        if self.with_neck:
            x_joint = self.neck(x_joint)
        
        # Split features back into RGB and HSI
        if hsi is not None:
            x_rgb = [f[:batch_size] for f in x_joint]
            x_hsi = [f[batch_size:] for f in x_joint]
        else:
            x_rgb = x_joint
            x_hsi = None

        losses = dict()
        
        # 2. RPN Forward and Loss (ONLY RGB)
        # In UDA, we only use labels for the source domain (RGB)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            # Use only x_rgb, img_metas, gt_bboxes
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_rgb, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # 3. ROI Forward and Loss (ONLY RGB)
        roi_losses = self.roi_head.forward_train(
            x_rgb, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        # 4. Cross-Modality Consistency Loss (Unsupervised Alignment)
        if hsi is not None and self.consistency_weight > 0:
            import torch.nn.functional as F
            consistency_loss = 0
            for feat_rgb, feat_hsi in zip(x_rgb, x_hsi):
                # We align x_hsi to x_rgb (which is supervised)
                consistency_loss += F.mse_loss(feat_hsi, feat_rgb.detach())
            losses['loss_consistency'] = consistency_loss * self.consistency_weight

        return losses
