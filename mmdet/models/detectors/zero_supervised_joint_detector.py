# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from .joint_modality_detector import JointModalityDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class ZeroSupervisedJointDetector(JointModalityDetector):
    """Zero-Supervised Joint Modality Detector.
    
    This detector ignores all bounding box and class labels. It is used
    solely for feature alignment between RGB and HSI domains.
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
        
        # 1. Extract joint features
        if hsi is not None:
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
        
        # 2. NO RPN or ROI Losses are calculated
        # We dummy-initialize some losses to prevent MMDetection from complaining about missing gradients
        # Actually, as long as loss_consistency has gradients, it should be fine.
        
        # 3. Cross-Modality Consistency Loss (The ONLY loss)
        if hsi is not None and self.consistency_weight > 0:
            consistency_loss = 0
            for feat_rgb, feat_hsi in zip(x_rgb, x_hsi):
                # Align x_hsi with x_rgb (both are updated only by consistency)
                # Note: per user request, we don't even use RGB labels, so both sides are equal.
                consistency_loss += F.mse_loss(feat_hsi, feat_rgb) 
            losses['loss_consistency'] = consistency_loss * self.consistency_weight

        # Add dummy losses with 0 weight if necessary, but usually not needed for training loop
        # losses['loss_dummy'] = (x_joint[0].sum() * 0)
        
        return losses
