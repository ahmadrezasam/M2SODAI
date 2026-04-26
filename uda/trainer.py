import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics.utils.nms import non_max_suppression
from uda.augmentations import weak_augmentation, strong_augmentation
from uda.pseudo_label import PseudoLabelFilter

class UDATrainer:
    """
    Simplified Mean Teacher UDA Trainer.
    - Student: Trainable model
    - Teacher: EMA of Student (non-trainable)
    """
    def __init__(self, student_model, source_loader, target_loader, val_loader, 
                 cfg=None):
        self.device = next(student_model.parameters()).device
        self.student = student_model
        
        # Teacher: EMA copy, no gradients
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.val_loader = val_loader
        
        # Default config
        self.cfg = {
            "epochs": 50,
            "ema_decay": 0.999,
            "lambda_pseudo": 1.0,
            "conf_thresh": 0.7,
            "imgsz": 640,
            "lr": 1e-4,
            "save_dir": "runs/uda_simple"
        }
        if cfg:
            self.cfg.update(cfg)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.cfg["lr"])
        self.pl_filter = PseudoLabelFilter(conf_thresh=self.cfg["conf_thresh"])
        
        # Augmentations (using Albumentations)
        self.weak_aug = weak_augmentation(self.cfg["imgsz"])
        self.strong_aug = strong_augmentation(self.cfg["imgsz"])

        import os
        os.makedirs(self.cfg["save_dir"], exist_ok=True)

    def _update_ema(self):
        """Update teacher weights using EMA of student weights."""
        decay = self.cfg["ema_decay"]
        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.data = decay * t.data + (1.0 - decay) * s.data

    def _apply_pseudo_label_loss(self, student_model, imgs, pseudo_labels):
        """
        Compute detection loss for pseudo-labels by formatting them as a YOLO batch.
        """
        if not any(len(p) > 0 for p in pseudo_labels):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Format for YOLO loss: 
        #   target concat: [batch_idx, cls, x, y, w, h] (normalized)
        #   img: [B, C, H, W]
        
        batch_idx_list = []
        cls_list = []
        bboxes_list = []
        
        # imgsz for normalization
        imgsz = imgs.shape[-1]
        
        for i, p in enumerate(pseudo_labels):
            if len(p) == 0:
                continue
            
            # [x1, y1, x2, y2, conf, cls] -> [x_c, y_c, w, h] normalized
            x1, y1, x2, y2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
            cls = p[:, 5]
            
            w = (x2 - x1) / imgsz
            h = (y2 - y1) / imgsz
            x = (x1 + x2) / 2 / imgsz
            y = (y1 + y2) / 2 / imgsz
            
            # Stack components
            batch_idx = torch.full((len(p), 1), i, device=self.device)
            target = torch.stack([batch_idx.squeeze(), cls, x, y, w, h], dim=1)
            
            batch_idx_list.append(batch_idx.squeeze())
            cls_list.append(cls)
            bboxes_list.append(torch.stack([x, y, w, h], dim=1))
            
        # Reconstruct batch dictionary
        # Ultralytics loss expects: batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx']
        target_batch = {
            'img': imgs,
            'cls': torch.cat(cls_list, dim=0).view(-1, 1),
            'bboxes': torch.cat(bboxes_list, dim=0),
            'batch_idx': torch.cat(batch_idx_list, dim=0)
        }
        
        # Compute loss
        # Note: self.student is exactly the LMW-YOLO DetectionModel
        # In YOLOv8/v11, forward(batch) returns loss if training=True
        loss, loss_items = student_model(target_batch)
        return loss.sum()

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()
        
        pbar = tqdm(zip(self.source_loader, self.target_loader), 
                    total=min(len(self.source_loader), len(self.target_loader)))
        
        for i, (batch_src, batch_tgt) in enumerate(pbar):
            # 2. Teacher generates pseudo-labels on Target (weakly augmented)
            # batch_tgt from Ultralytics has 'img' (B, 3, H, W) in [0, 255]
            with torch.no_grad():
                tgt_weak = batch_tgt["img"].to(self.device).float() / 255.0
                raw_preds = self.teacher(tgt_weak)
                pseudo = self.pl_filter.filter(raw_preds)
            
            # 3. Student on source (supervised) and target (unsupervised)
            # Source loss
            for k, v in batch_src.items():
                if isinstance(v, torch.Tensor):
                    batch_src[k] = v.to(self.device)
            loss_src, _ = self.student(batch_src)
            loss_src = loss_src.sum()
            
            # Target (pseudo-label) loss
            # For simplicity, we use same images for now, but ideally apply strong aug
            tgt_strong = batch_tgt["img"].to(self.device).float() / 255.0 # TODO: strong aug
            loss_pseudo = self._apply_pseudo_label_loss(self.student, tgt_strong, pseudo)

            # 4. Total Loss
            loss = loss_src + self.cfg["lambda_pseudo"] * loss_pseudo
            
            # 5. Backward and Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 6. EMA Update
            self._update_ema()
            
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    def train(self):
        for epoch in range(self.cfg["epochs"]):
            self.train_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                self.save(epoch + 1)
    
    def save(self, epoch):
        path = f"{self.cfg['save_dir']}/teacher_epoch_{epoch}.pt"
        torch.save(self.teacher.state_dict(), path)
        print(f"Saved teacher checkpoint to {path}")
