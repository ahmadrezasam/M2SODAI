import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Gradient Reversal Layer (GRL)
# ==========================================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

# ==========================================
# 2. Spectral-Specific Architecture (3D CNN)
# ==========================================
# Instead of a standard 2D CNN (YOLO), we use a 3D CNN specifically designed for
# HSI/Multispectral images. 3D CNNs treat the spectral bands as depth, 
# capturing joint spatial-spectral features.
class Spectral3DCNN(nn.Module):
    def __init__(self, in_channels=30, out_features=256):
        super().__init__()
        # Input shape: (Batch, 1, Channels (30), H, W) -> Notice we add a '1' dimension for 3D conv
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(16)
        
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(32)

        # We compress the remaining spectral dimension using a 1x1 2D Conv
        # to match the YOLO spatial features.
        self.spatial_compress = nn.Conv2d(in_channels=32 * 18, out_channels=out_features, kernel_size=1)

    def forward(self, x):
        # x is (B, 30, H, W) -> Reshape to (B, 1, 30, H, W) for 3D Conv
        x = x.unsqueeze(1)
        
        x = self.relu(self.bn1(self.conv1(x))) # e.g., Depth goes from 30 -> 24
        x = self.relu(self.bn2(self.conv2(x))) # e.g., Depth goes from 24 -> 20
        x = self.relu(self.bn3(self.conv3(x))) # e.g., Depth goes from 20 -> 18
        
        # Flatten the channel and spectral dimensions: (B, 32, 18, H, W) -> (B, 32*18, H, W)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        
        # Project down to the fusion feature size
        x = self.spatial_compress(x)
        return x

# ==========================================
# 3. Spatial Mock Architecture (YOLO backbone representation)
# ==========================================
class SpatialYOLOMock(nn.Module):
    def __init__(self, in_channels=3, out_features=256):
        super().__init__()
        # A mock of YOLO's spatial feature extraction (Darknet/CSP layers)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, out_features, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

# ==========================================
# 4. Dual-Branch Network & Domain Discriminator
# ==========================================
class DualBranchUDA(nn.Module):
    def __init__(self, hsi_channels=30, rgb_channels=3, feature_dim=256, num_classes=1):
        super().__init__()
        # 1. The Branches
        self.spatial_branch = SpatialYOLOMock(in_channels=rgb_channels, out_features=feature_dim)
        self.spectral_branch = Spectral3DCNN(in_channels=hsi_channels, out_features=feature_dim)
        
        # 2. Fusion (Concatenate and project)
        self.fusion = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)
        
        # 3. Task Head (Mock YOLO OBB Head)
        self.task_head = nn.Conv2d(feature_dim, num_classes * 5, kernel_size=1) # 5 for OBB (x, y, w, h, angle)
        
        # 4. Domain Discriminator (For Adversarial UDA)
        self.grl = GRL(alpha=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Outputs 0 for Source (RGB), 1 for Target (HSI)
        )

    def forward(self, rgb=None, hsi=None, alpha=1.0):
        # We need to process whatever is passed in. 
        # In a real batch, if rgb is passed, we might just pass zeros to spectral, 
        # or if we have both, we use both.
        
        # For this mock, we assume if you pass RGB, you are training the Spatial branch.
        # If you pass HSI, you are training the Spectral branch.
        
        # Extract Spatial Features
        if rgb is not None:
            spatial_feat = self.spatial_branch(rgb)
        else:
            # Dummy spatial features if only HSI is provided
            spatial_feat = torch.zeros(hsi.size(0), 256, hsi.size(2)//8, hsi.size(3)//8, device=hsi.device)
            
        # Extract Spectral Features
        if hsi is not None:
            spectral_feat = self.spectral_branch(hsi)
            # We need to pool or interpolate to match spatial resolution for fusion
            spectral_feat = F.interpolate(spectral_feat, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False)
        else:
            # Dummy spectral features if only RGB is provided
            spectral_feat = torch.zeros_like(spatial_feat)
            
        # FUSION
        fused_feat = torch.cat([spatial_feat, spectral_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        # TASK PREDICTION (e.g., Object Detection)
        task_out = self.task_head(fused_feat)
        
        # DOMAIN DISCRIMINATION (Adversarial)
        # Apply GRL before the discriminator
        grl_feat = self.grl(fused_feat)
        self.grl.alpha = alpha # Update alpha dynamically
        domain_out = self.domain_discriminator(grl_feat)
        
        return task_out, domain_out

# ==========================================
# 5. Training Loop (Simplest Mean Teacher + Adv)
# ==========================================
def update_ema_variables(model, ema_model, alpha=0.999):
    # Update Teacher weights via Exponential Moving Average
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_mock():
    print("Initializing Dual-Branch UDA Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Student and Teacher
    student = DualBranchUDA().to(device)
    teacher = DualBranchUDA().to(device)
    teacher.load_state_dict(student.state_dict()) # Initial sync
    
    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    
    # Losses
    bce_loss = nn.BCELoss() # For Domain Discriminator
    mse_loss = nn.MSELoss() # For Mean Teacher consistency
    
    print("\nStarting Mock Training Loop...")
    for step in range(5):
        optimizer.zero_grad()
        
        # ---------------------------------------------
        # 1. Source Domain (Labeled RGB)
        # ---------------------------------------------
        # Dummy batch of 4 RGB images (e.g., aerial photography)
        source_rgb = torch.randn(4, 3, 256, 256).to(device) 
        source_labels = torch.randn(4, 5, 32, 32).to(device) # Mock YOLO targets
        
        # Forward pass Source
        src_task_pred, src_domain_pred = student(rgb=source_rgb, hsi=None)
        
        # Source Task Loss (Supervised YOLO loss mock)
        loss_task = mse_loss(src_task_pred, source_labels)
        
        # Source Domain Loss (Discriminator should predict 0 for Source)
        src_domain_labels = torch.zeros_like(src_domain_pred)
        loss_domain_src = bce_loss(src_domain_pred, src_domain_labels)
        
        # ---------------------------------------------
        # 2. Target Domain (Unlabeled HSI)
        # ---------------------------------------------
        # Dummy batch of 4 HSI images (30 channels)
        target_hsi = torch.randn(4, 30, 256, 256).to(device)
        
        # Forward pass Target (Student)
        tgt_task_pred, tgt_domain_pred = student(rgb=None, hsi=target_hsi)
        
        # Target Domain Loss (Discriminator should predict 1 for Target)
        tgt_domain_labels = torch.ones_like(tgt_domain_pred)
        loss_domain_tgt = bce_loss(tgt_domain_pred, tgt_domain_labels)
        
        # ---------------------------------------------
        # 3. Mean Teacher Consistency (Unlabeled HSI)
        # ---------------------------------------------
        with torch.no_grad():
            # In a real scenario, you pass a weakly augmented target to teacher
            # and a strongly augmented target to student.
            tgt_teacher_pred, _ = teacher(rgb=None, hsi=target_hsi)
            
        loss_consistency = mse_loss(tgt_task_pred, tgt_teacher_pred)
        
        # ---------------------------------------------
        # 4. Total Loss & Backward Pass
        # ---------------------------------------------
        # The GRL handles reversing the gradients from the discriminator to the backbone,
        # forcing the backbone to extract domain-invariant features.
        loss_total = loss_task + (loss_domain_src + loss_domain_tgt) * 0.1 + loss_consistency * 0.1
        
        loss_total.backward()
        optimizer.step()
        
        # Update Teacher
        update_ema_variables(student, teacher, alpha=0.999)
        
        print(f"Step {step+1}/5 | Total Loss: {loss_total.item():.4f} "
              f"| Task: {loss_task.item():.4f} "
              f"| Domain (Src): {loss_domain_src.item():.4f} "
              f"| Consistency: {loss_consistency.item():.4f}")

    print("\nMock training completed successfully! No crashes.")

if __name__ == "__main__":
    train_mock()
