# M²SODAI — Method & Model: Detailed Explanation

## 1. Problem Formulation

This is a **two-class object detection** task on co-registered multi-modal imagery:

- **Input:** An RGB image $I_{RGB} \in \mathbb{R}^{H_r \times W_r \times 3}$ (resized to $1600 \times 1600$) and a hyperspectral (HSI) cube $I_{HSI} \in \mathbb{R}^{H_h \times W_h \times 30}$ (resized to $224 \times 224$), where the 30 channels come from PCA dimensionality reduction of the original spectral bands.
- **Output:** A set of bounding boxes with class labels: **ship** or **floatingmatter**.

---

## 2. Overall Architecture: `FasterRCNNDFPN` (Dual-FPN Faster R-CNN)

The model is defined as `TwoStageDetectorDFPN` in `mmdet/models/detectors/two_stage_DFPN.py`. It is a **dual-stream two-stage detector** with the following data flow:

```
                        ┌────────────────────────────────────────────────┐
                        │            RGB Stream                          │
  RGB (1600×1600×3)  ──>│  ResNet-50 (pretrained) ──> 4 stage outputs   │──┐
                        └────────────────────────────────────────────────┘  │
                                                                            │
                        ┌────────────────────────────────────────────────┐  │  Attention-Gated
                        │            HSI Stream                          │  ├─ Fusion (per-level)
  HSI (224×224×30)   ──>│  3D Conv Stem ──> ResNet-50 ──> FPN_hsi       │──┘        │
                        └────────────────────────────────────────────────┘          │
                                                                                    ▼
                                                                             Fused Features
                                                                                    │
                                                                               FPN (main)
                                                                                    │
                                                                    ┌───────────────┴───────────────┐
                                                                    ▼                               ▼
                                                                RPN Head                      RoI Head
                                                             (region proposals)         (bbox cls + reg)
```

---

## 3. Component-by-Component Breakdown

### 3.1 RGB Backbone — ResNet-50

Defined in the config at `configs/faster_rcnn/faster_rcnn_r50_rgb_hsi.py` (lines 16–29):

- Standard **ResNet-50** with 4 stages
- **Pretrained** from `torchvision://resnet50` (ImageNet weights)
- Uses **Group Normalization** (GN, 32 groups) instead of BatchNorm — this is critical because the batch size is 1, where BN statistics are unreliable
- `frozen_stages=-1` means **no layers are frozen** — the entire backbone is fine-tuned
- `with_cp=True` enables **gradient checkpointing** on residual blocks to reduce VRAM
- Outputs feature maps at 4 scales: $C_1, C_2, C_3, C_4$ with channel counts `[256, 512, 1024, 2048]`

### 3.2 HSI Backbone — Modified ResNet-50

Defined at lines 30–44 of the same config, with crucial modifications applied in `__init__` of `TwoStageDetectorDFPN` (`two_stage_DFPN.py`, lines 129–130):

```python
self.backbone_hsi = build_backbone(backbone_hsi)
self.backbone_hsi.conv1 = conv1_hsi()       # Replace standard 7×7 conv with 3D conv stem
self.backbone_hsi.maxpool = nn.Identity()    # Remove max pooling
```

- **No pretrained weights** (`init_cfg=None`) — trained from scratch since there are no pretrained models for 30-channel HSI
- The standard ResNet `conv1` (7×7, stride 2, 3→64 channels) is **entirely replaced** by the custom `conv1_hsi` module
- The `maxpool` after conv1 is replaced with `nn.Identity()` to preserve spatial resolution in the low-resolution HSI input
- HSI input is preprocessed with `tanh()` before entering: `x_hsi = self.backbone_hsi(hsi.tanh())` — this bounds HSI values to $[-1, 1]$

### 3.3 The 3D Conv Stem (`conv1_hsi`) — Multi-Scale Spectral Feature Extraction

This is the most novel component (`two_stage_DFPN.py`, lines 55–103). It processes the **spectral dimension** using 3D convolutions before feeding into the 2D ResNet. It consists of **three parallel branches** operating at different spectral receptive fields:

**Branch 1 (`nets1`) — Large spectral kernel:**

$$\text{Conv3D}(1 \to 6,\ k{=}7^3,\ s{=}(2,1,1)) \to \text{Mish} \to \text{Conv3D}(6 \to 12,\ k{=}3^3,\ s{=}(2,1,1)) \to \text{Mish} \to \text{Flatten}$$

- Two layers of 3D conv, stride 2 only in spectral dimension
- Reduces 30 spectral bands → ~7 → final flattened channel output

**Branch 2 (`nets2`) — Medium spectral kernel:**

$$\text{Conv3D}(1 \to 6,\ k{=}5^3) \to \text{Mish} \to \text{Conv3D}(6 \to 12,\ k{=}3^3) \to \text{Mish} \to \text{Conv3D}(12 \to 24,\ k{=}3^3) \to \text{Mish} \to \text{Flatten}$$

- Three layers of 3D conv with progressively increasing channel width

**Branch 3 (`nets3`) — Small spectral kernel:**

$$\text{Conv3D}(1 \to 6,\ k{=}3^3) \to \text{Mish} \to \text{Conv3D}(6 \to 12) \to \text{Mish} \to \text{Conv3D}(12 \to 24) \to \text{Mish} \to \text{Conv3D}(24 \to 48) \to \text{Mish} \to \text{Flatten}$$

- Four layers of 3D conv — deepest branch

**Fusion:** The three branches are concatenated along the channel dimension (producing $84 + 96 + 96 = 276$ channels), then reduced to 64 channels via a $3 \times 3$ 2D convolution:

$$\text{output} = \text{Conv2D}_{3\times3}\big(\text{cat}[\text{branch}_1,\ \text{branch}_2,\ \text{branch}_3]\big) \in \mathbb{R}^{B \times 64 \times H \times W}$$

The **Mish activation** is used throughout:

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))$$

which is a smooth, non-monotonic activation that has shown benefits over ReLU in certain architectures.

**Design rationale:** Different branches capture spectral features at different granularities — from broad spectral trends (branch 1, large kernel) to fine spectral details (branch 3, small kernel stacked deeply). This is analogous to an Inception-style multi-scale processing, but applied to the spectral dimension.

### 3.4 HSI Feature Pyramid Network (`neck_hsi`)

Defined at lines 54–60 of the config:

- Standard **FPN** taking ResNet-50 outputs `[256, 512, 1024, 2048]` → 256 channel outputs at 5 scales
- Group Normalization
- This FPN's output serves **two purposes**:
  1. Generates the **attention maps** for fusion
  2. Provides spatially-enhanced HSI features

### 3.5 Attention-Gated Fusion

The key fusion mechanism is in `_fuse_level` and `extract_feat` (`two_stage_DFPN.py`, lines 170–201):

```python
def _fuse_level(self, x_i, x_hsi_i, x_att_i):
    h, w = x_i.shape[-2:]
    hsi_i = F.interpolate(x_hsi_i, size=(h, w), mode='bilinear')
    att_i = self.conv_att(x_att_i)          # Conv2d(256→1) + Sigmoid
    att_i = F.interpolate(att_i, size=(h, w), mode='bilinear')
    return att_i * torch.cat((x_i, hsi_i), dim=1)
```

For each FPN level $i$, the fusion computes:

$$\alpha_i = \sigma\Big(\text{Conv}_{1\times1}\big(F_{HSI}^{(i)}\big)\Big) \in [0, 1]^{H_i \times W_i}$$

$$F_{fused}^{(i)} = \alpha_i \odot \Big[F_{RGB}^{(i)}\ \|\ \text{Interp}(F_{HSI\_backbone}^{(i)})\Big]$$

Where:
- $F_{HSI}^{(i)}$ = HSI FPN output at level $i$ (256 channels)
- $\alpha_i$ = spatial attention gate (1 channel, sigmoid-activated)
- $F_{RGB}^{(i)}$ = RGB backbone output at stage $i$ (before the main FPN)
- $\text{Interp}$ = bilinear interpolation from HSI resolution to RGB resolution
- $\|$ = channel-wise concatenation
- $\odot$ = element-wise (broadcast) multiplication

The **attention map** $\alpha_i$ learns to gate which spatial locations benefit from multi-modal information vs. which should be suppressed. The **same `conv_att` module** (a single, shared $1 \times 1$ conv + sigmoid) is used across all FPN levels — this acts as a lightweight channel-to-spatial attention.

**Critical note on channel doubling:** The concatenation of RGB backbone features (e.g., 256 channels at stage 1) with HSI backbone features (also 256 at the same stage) **doubles the channel count**. This explains why the main FPN has `in_channels=[512, 1024, 2048, 4096]` instead of the standard `[256, 512, 1024, 2048]`.

### 3.6 Main FPN (RGB-HSI Fused)

Defined at lines 45–53 of the config:

- Takes the **fused** feature maps: `[512, 1024, 2048, 4096]` → 256 channels, 5 output levels
- Output strides: [4, 8, 16, 32, 64]
- The top-down pathway with lateral connections provides multi-scale feature maps for detection

### 3.7 Region Proposal Network (RPN)

Standard Faster R-CNN RPN (config lines 62–78):

| Parameter | Value |
|---|---|
| Anchor scales | [8] |
| Anchor ratios | [0.5, 1.0, 2.0] |
| Strides | [4, 8, 16, 32, 64] |
| Positive IoU threshold | 0.7 |
| Negative IoU threshold | 0.3 |
| Samples per image | 256 (50% positive) |
| Train NMS top-k | 2000 pre-NMS, 1000 post-NMS |
| Test NMS IoU threshold | 0.9 (relaxed to keep more proposals) |
| Loss | CrossEntropy (cls) + SmoothL1 (reg) |

With a 1600×1600 input and strides [4, 8, 16, 32, 64], this produces anchor grids of sizes $400^2 + 200^2 + 100^2 + 50^2 + 25^2$ ≈ 200K+ anchor locations, each with 3 aspect ratios = **~640K total anchors**.

### 3.8 RoI Head — `Shared4Conv1FCBBoxHead`

Defined at config lines 80–100 and implemented in `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`:

The RoI features are extracted via **RoIAlign** (7×7 output, from strides [4, 8, 16, 32]) and then processed by:

```
RoIAlign 7×7
    │
    ▼
Conv 3×3 (256→256, GN) ──> ReLU
Conv 3×3 (256→256, GN) ──> ReLU
Conv 3×3 (256→256, GN) ──> ReLU
Conv 3×3 (256→256, GN) ──> ReLU
    │
    ▼
Flatten  →  FC (256×7×7=12544 → 1024)  →  ReLU
                    │                          │
                    ▼                          ▼
              FC cls (1024 → 3)         FC reg (1024 → 8)
              [bg, ship, float]         [dx,dy,dw,dh × 2 classes]
```

- 4 shared convolutional layers (each 3×3, 256 channels, GN normalized)
- 1 shared FC layer (12544 → 1024)
- Classification: 3-way softmax (background + 2 classes)
- Regression: **class-specific** (`reg_class_agnostic=False`), so 4 box deltas per class = 8 outputs
- Loss: CrossEntropy (cls) + SmoothL1 (bbox reg)
- Test-time NMS: **Soft-NMS** (IoU threshold 0.5, min score 0.01) — softer than standard NMS, better for overlapping maritime objects

---

## 4. Data Pipeline

### 4.1 Loading (`LoadImageFrom_JPG_HSI`)

For each sample, both modalities are loaded:
1. **RGB:** Standard JPEG loaded via `mmcv.imfrombytes` → `uint8` array in BGR order
2. **HSI:** Corresponding `.mat` file loaded via `scipy.io.loadmat` → `float16` array with 30 PCA-reduced channels, Z-score normalized:

$$(x - \mu_{PCA}) / \sigma_{PCA}$$

### 4.2 Augmentation & Preprocessing

| Step | RGB | HSI |
|---|---|---|
| `RandomFlip_JPG_HSI` | Horizontal flip (p=0.75) | Same flip applied |
| `Resize_JPG_HSI` | 1600×1600, keep ratio | 224×224, keep ratio |
| `Normalize_JPG_HSI` | mean=[123.6, 116.2, 103.5], std=[58.39, 56.12, 57.3] | Already PCA-normalized at load time |
| `Pad` | Pad to multiple of 32 | — |

The **synchronized flip** ensures spatial correspondence between modalities is preserved.

---

## 5. Training Schedule

| Phase | Epochs | Learning Rate |
|---|---|---|
| Warm-up | ~0.6 epochs (500 iters) | 0.002 → 0.02 (linear) |
| Base training | 1–65 | 0.02 |
| Decay 1 | 65–71 | 0.002 (÷10) |
| Decay 2 | 71–73 | 0.0002 (÷10) |

- **Mixed precision (fp16)** with loss scaling of 512 to prevent gradient underflow
- **No gradient clipping** — relies on GN stability
- **Norm decay = 0** — GN parameters (γ, β) are not weight-decayed

---

## 6. Key Design Decisions & Rationale

1. **Asymmetric resolutions (1600 vs 224):** HSI sensors have inherently lower spatial resolution. Rather than upsample HSI to match RGB, the model processes each at its native resolution and fuses at the feature level via bilinear interpolation.

2. **3D conv stem vs. simple channel projection:** A naïve approach would be a single 2D conv to reduce 30→64 channels. The 3D conv stem captures **spectral correlations** across adjacent bands — spectral information that a single 2D conv cannot exploit.

3. **Attention gating:** Rather than simply concatenating RGB and HSI features (which would treat both modalities equally everywhere), the learned spatial attention $\alpha$ allows the model to **selectively rely on HSI** in regions where spectral information is discriminative (e.g., floating matter with distinct spectral signature) while **downweighting it** where it adds noise.

4. **Fusion before the main FPN, not after:** The fused features pass through a full FPN with top-down connections, allowing high-level semantic information to propagate back to fine-grained spatial levels. This is more powerful than late fusion at the detection head.

5. **Group Normalization everywhere:** With batch size 1 (forced by the dual-stream memory footprint), BatchNorm would produce degenerate statistics. GN normalizes over groups of channels within each sample, independent of batch size.
