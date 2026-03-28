# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector
from ..utils.spectral_dropout import SpectralDropout


@DETECTORS.register_module()
class MeanTeacherUDADetector(TwoStageDetector):
    """Mean Teacher UDA Detector with Instance-Level Adversarial Alignment
    and Spectral Dropout.

    This detector implements a 3-component UDA system:
    1. **Mean Teacher**: EMA teacher generates pseudo-labels on HSI (target)
    2. **Instance-Level Adversarial**: Discriminator on RoI features with GRL
    3. **Spectral Dropout**: Random band zeroing for regularization

    Training:
        - RGB (source): supervised detection losses
        - HSI (target): pseudo-label losses from teacher + adversarial alignment
        - Burn-in: first N epochs, skip pseudo-label loss

    Inference:
        - HSI only → projector → backbone → FPN → RPN → ROI → detections

    Args:
        backbone (dict): Backbone config (ProjectUpsampleResNet).
        neck (dict): Neck config (FPN).
        rpn_head (dict): RPN head config.
        roi_head (dict): ROI head config.
        instance_discriminator (dict): Instance-level domain discriminator cfg.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
        ema_decay (float): EMA decay rate for teacher. Default: 0.9996.
        pseudo_thr (float): Initial pseudo-label confidence threshold.
            Default: 0.9.
        pseudo_thr_min (float): Minimum pseudo-label threshold (decays to).
            Default: 0.7.
        pseudo_warmup_epochs (int): Epochs over which threshold decays.
            Default: 10.
        burn_in_epochs (int): Epochs before enabling pseudo-label loss.
            Default: 5.
        pseudo_loss_weight (float): Weight for pseudo-label loss. Default: 1.0.
        domain_loss_weight (float): Weight for adversarial loss. Default: 0.1.
        spectral_dropout_rate (float): Band dropout rate. Default: 0.3.
        pretrained (str): Pretrained model path.
        init_cfg (dict): Initialization config.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 instance_discriminator=None,
                 train_cfg=None,
                 test_cfg=None,
                 ema_decay=0.9996,
                 pseudo_thr=0.9,
                 pseudo_thr_min=0.7,
                 pseudo_warmup_epochs=10,
                 burn_in_epochs=5,
                 pseudo_loss_weight=1.0,
                 domain_loss_weight=0.1,
                 spectral_dropout_rate=0.3,
                 pretrained=None,
                 init_cfg=None):
        super(MeanTeacherUDADetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # --- Mean Teacher ---
        self.ema_decay = ema_decay
        self.pseudo_thr = pseudo_thr
        self.pseudo_thr_min = pseudo_thr_min
        self.pseudo_warmup_epochs = pseudo_warmup_epochs
        self.burn_in_epochs = burn_in_epochs
        self.pseudo_loss_weight = pseudo_loss_weight

        # Build teacher as a deep copy (will be updated via EMA)
        self._build_teacher(backbone, neck, rpn_head, roi_head, 
                           train_cfg, test_cfg)

        # --- Instance-Level Adversarial ---
        self.domain_loss_weight = domain_loss_weight
        if instance_discriminator is not None:
            self.instance_discriminator = build_neck(instance_discriminator)
        else:
            self.instance_discriminator = None

        # --- Spectral Dropout ---
        self.spectral_dropout = SpectralDropout(p=spectral_dropout_rate)

        # Track current epoch for adaptive thresholding
        self._current_epoch = 0
        # Per-class threshold tracking (will be initialized on first use)
        self._class_thresholds = None

    def _build_teacher(self, backbone, neck, rpn_head, roi_head,
                       train_cfg, test_cfg):
        """Build teacher model as a complete copy of the student."""
        self.teacher_backbone = build_backbone(backbone)
        if neck is not None:
            self.teacher_neck = build_neck(neck)
        else:
            self.teacher_neck = None
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = copy.deepcopy(rpn_head)
            rpn_head_.update(train_cfg=rpn_train_cfg, 
                            test_cfg=test_cfg.rpn if test_cfg is not None else None)
            self.teacher_rpn = build_head(rpn_head_)
        else:
            self.teacher_rpn = None
        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head_ = copy.deepcopy(roi_head)
            roi_head_.update(train_cfg=rcnn_train_cfg,
                            test_cfg=test_cfg.rcnn if test_cfg is not None else None)
            self.teacher_roi = build_head(roi_head_)
        else:
            self.teacher_roi = None

        # Freeze teacher parameters
        for param in self._teacher_parameters():
            param.requires_grad = False

    def _teacher_parameters(self):
        """Yield all teacher parameters."""
        yield from self.teacher_backbone.parameters()
        if self.teacher_neck is not None:
            yield from self.teacher_neck.parameters()
        if self.teacher_rpn is not None:
            yield from self.teacher_rpn.parameters()
        if self.teacher_roi is not None:
            yield from self.teacher_roi.parameters()

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher weights via EMA from student weights."""
        alpha = self.ema_decay
        # Backbone
        for t_param, s_param in zip(self.teacher_backbone.parameters(),
                                     self.backbone.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
        # Neck
        if self.teacher_neck is not None and self.neck is not None:
            for t_param, s_param in zip(self.teacher_neck.parameters(),
                                         self.neck.parameters()):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
        # RPN
        if self.teacher_rpn is not None and self.rpn_head is not None:
            for t_param, s_param in zip(self.teacher_rpn.parameters(),
                                         self.rpn_head.parameters()):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
        # ROI
        if self.teacher_roi is not None and self.roi_head is not None:
            for t_param, s_param in zip(self.teacher_roi.parameters(),
                                         self.roi_head.parameters()):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    def _get_current_threshold(self):
        """Compute adaptive pseudo-label threshold based on current epoch."""
        if self._current_epoch >= self.pseudo_warmup_epochs:
            return self.pseudo_thr_min
        # Linear decay from pseudo_thr to pseudo_thr_min
        progress = self._current_epoch / max(self.pseudo_warmup_epochs, 1)
        return self.pseudo_thr - (self.pseudo_thr - self.pseudo_thr_min) * progress

    @torch.no_grad()
    def generate_pseudo_labels(self, hsi, img_metas):
        """Generate pseudo-labels on HSI using the teacher model.

        Args:
            hsi (Tensor): HSI input (B, C_hsi, H_hsi, W_hsi).
            img_metas (list[dict]): Meta info.

        Returns:
            tuple: (pseudo_bboxes, pseudo_labels) - lists of tensors per image.
                Returns empty tensors if no confident pseudo-labels found.
        """
        # Teacher forward on HSI
        x = self.teacher_backbone(hsi)
        if self.teacher_neck is not None:
            x = self.teacher_neck(x)

        # RPN proposals
        proposal_list = self.teacher_rpn.simple_test_rpn(x, img_metas)
        # ROI predictions
        results = self.teacher_roi.simple_test(
            x, proposal_list, img_metas, rescale=False)

        thresh = self._get_current_threshold()
        pseudo_bboxes = []
        pseudo_labels = []

        for result in results:
            # result is list of arrays per class, shape (N, 5) with scores
            bboxes_list = []
            labels_list = []
            for cls_id, cls_result in enumerate(result):
                if len(cls_result) == 0:
                    continue
                scores = cls_result[:, 4]
                keep = scores >= thresh
                if keep.any():
                    kept_bboxes = cls_result[keep, :4]
                    kept_labels = [cls_id] * int(keep.sum())
                    bboxes_list.append(
                        torch.tensor(kept_bboxes, dtype=torch.float32,
                                     device=hsi.device))
                    labels_list.append(
                        torch.tensor(kept_labels, dtype=torch.long,
                                     device=hsi.device))

            if bboxes_list:
                pseudo_bboxes.append(torch.cat(bboxes_list, dim=0))
                pseudo_labels.append(torch.cat(labels_list, dim=0))
            else:
                pseudo_bboxes.append(
                    torch.zeros((0, 4), dtype=torch.float32, device=hsi.device))
                pseudo_labels.append(
                    torch.zeros((0,), dtype=torch.long, device=hsi.device))

        return pseudo_bboxes, pseudo_labels

    def _extract_rgb_feat(self, img):
        """Extract features from RGB using the student backbone (bypass projector)."""
        if hasattr(self.backbone, 'resnet'):
            x = self.backbone.resnet(img)
        else:
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _extract_hsi_feat(self, hsi):
        """Extract features from HSI using the full student pipeline."""
        # Apply spectral dropout before projector
        hsi = self.spectral_dropout(hsi)
        # Full backbone forward: projector → upsample → resnet
        x = self.backbone(hsi)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _get_roi_feats(self, x, proposals):
        """Extract RoI features for adversarial alignment.

        Args:
            x (list[Tensor]): Multi-level features from FPN.
            proposals (list[Tensor]): Proposals per image.

        Returns:
            Tensor: RoI features of shape (N_total_rois, C, H, W).
        """
        rois = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs],
            proposals)
        return rois

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
        """Forward training.

        Args:
            img (Tensor): RGB source images (B, 3, H, W).
            img_metas (list[dict]): Image meta info.
            gt_bboxes (list[Tensor]): GT bboxes for RGB.
            gt_labels (list[Tensor]): GT labels for RGB.
            hsi (Tensor, optional): HSI target images (B, C_hsi, H, W).
        """
        losses = dict()

        # ============ 1. Supervised Detection on RGB (Source) ============
        x_rgb = self._extract_rgb_feat(img)

        # RPN
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                               self.test_cfg.rpn)
            rpn_losses, proposal_list_rgb = self.rpn_head.forward_train(
                x_rgb, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg, **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list_rgb = proposals

        # ROI
        roi_losses = self.roi_head.forward_train(
            x_rgb, img_metas, proposal_list_rgb, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        # ============ 2. HSI Branch ============
        if hsi is not None:
            # Extract HSI features (with spectral dropout)
            x_hsi = self._extract_hsi_feat(hsi)

            # --- 2a. Pseudo-Label Loss (after burn-in) ---
            if self._current_epoch >= self.burn_in_epochs:
                pseudo_bboxes, pseudo_labels = self.generate_pseudo_labels(
                    hsi, img_metas)

                # Count how many pseudo-labels we have  
                n_pseudo = sum(len(pb) for pb in pseudo_bboxes)

                if n_pseudo > 0:
                    # Generate proposals on HSI student features for pseudo loss
                    proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                       self.test_cfg.rpn)
                    _, proposal_list_hsi = self.rpn_head.forward_train(
                        x_hsi, img_metas, pseudo_bboxes, gt_labels=None,
                        gt_bboxes_ignore=None,
                        proposal_cfg=proposal_cfg)

                    # ROI loss on pseudo-labels
                    pseudo_roi_losses = self.roi_head.forward_train(
                        x_hsi, img_metas, proposal_list_hsi,
                        pseudo_bboxes, pseudo_labels,
                        gt_bboxes_ignore=None, gt_masks=None)

                    for key, val in pseudo_roi_losses.items():
                        losses[f'pseudo_{key}'] = \
                            val * self.pseudo_loss_weight \
                            if not isinstance(val, list) \
                            else [v * self.pseudo_loss_weight for v in val]

                    losses['n_pseudo_labels'] = torch.tensor(
                        float(n_pseudo), device=img.device)

            # --- 2b. Instance-Level Adversarial Loss ---
            if self.instance_discriminator is not None:
                # Get RoI features from both domains
                # Use detached proposals to avoid adversarial signal in RPN
                with torch.no_grad():
                    proposal_cfg = self.test_cfg.rpn
                    proposals_rgb_det = self.rpn_head.simple_test_rpn(
                        x_rgb, img_metas)
                    proposals_hsi_det = self.rpn_head.simple_test_rpn(
                        x_hsi, img_metas)

                # Convert proposals to rois format
                from mmdet.core import bbox2roi
                rois_rgb = bbox2roi(proposals_rgb_det)
                rois_hsi = bbox2roi(proposals_hsi_det)

                if len(rois_rgb) > 0 and len(rois_hsi) > 0:
                    # Extract RoI features
                    roi_feats_rgb = self.roi_head.bbox_roi_extractor(
                        x_rgb[:self.roi_head.bbox_roi_extractor.num_inputs],
                        rois_rgb)
                    roi_feats_hsi = self.roi_head.bbox_roi_extractor(
                        x_hsi[:self.roi_head.bbox_roi_extractor.num_inputs],
                        rois_hsi)

                    # Limit number of RoIs to avoid OOM
                    max_rois = 128
                    if roi_feats_rgb.shape[0] > max_rois:
                        idx = torch.randperm(
                            roi_feats_rgb.shape[0])[:max_rois]
                        roi_feats_rgb = roi_feats_rgb[idx]
                    if roi_feats_hsi.shape[0] > max_rois:
                        idx = torch.randperm(
                            roi_feats_hsi.shape[0])[:max_rois]
                        roi_feats_hsi = roi_feats_hsi[idx]

                    # Domain predictions (GRL is inside discriminator)
                    domain_pred_rgb = self.instance_discriminator(
                        roi_feats_rgb)
                    domain_pred_hsi = self.instance_discriminator(
                        roi_feats_hsi)

                    # Domain labels: RGB=0, HSI=1
                    label_rgb = torch.zeros_like(domain_pred_rgb)
                    label_hsi = torch.ones_like(domain_pred_hsi)

                    loss_domain = (
                        F.binary_cross_entropy_with_logits(
                            domain_pred_rgb, label_rgb) +
                        F.binary_cross_entropy_with_logits(
                            domain_pred_hsi, label_hsi)
                    ) / 2.0

                    losses['loss_domain'] = loss_domain * self.domain_loss_weight

        # ============ 3. Update Teacher (EMA) ============
        self.update_teacher()

        return losses

    def simple_test(self, img, img_metas, hsi=None, proposals=None,
                    rescale=False):
        """Test on HSI (target domain) or RGB (source domain).

        At inference, we always evaluate on HSI through the full pipeline.
        """
        if hsi is not None:
            if isinstance(hsi, list):
                hsi = hsi[0]
            # Use student model for inference on HSI
            x = self.backbone(hsi)
            if self.with_neck:
                x = self.neck(x)
        else:
            if isinstance(img, list):
                img = img[0]
            if hasattr(self.backbone, 'resnet'):
                x = self.backbone.resnet(img)
            else:
                x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)

        if isinstance(img_metas, list) and \
                len(img_metas) > 0 and isinstance(img_metas[0], list):
            img_metas = img_metas[0]

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def set_epoch(self, epoch):
        """Called by the training hook to update the current epoch."""
        self._current_epoch = epoch

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
