# JOHDA: Joint Optical-Hyperspectral Domain Adaptation Architecture

This document describes the internal architecture of the `JointModalityDetector` and the Cross-Modality Consistency (CMC) training strategy.

## 1. High-Level Concept
JOHDA adapts a pre-trained RGB object detector to the hyperspectral (HSI) domain by training on co-registered image pairs. It uses a shared backbone and neck to align the features of both modalities.

```mermaid
graph TD
    subgraph "Input Modalities (Co-registered)"
        RGB["RGB Image (3x1600x1600)"]
        HSI["HSI Image (30x224x224)"]
    end

    subgraph "HSI Adaptation Front-end"
        Projector["1x1 Conv Projector (30 -> 3 bands)"]
        Upsample["Bilinear Upsampling (224 -> 1600)"]
    end

    subgraph "Shared Detector 'Brain'"
        Backbone["Shared ResNet Backbone (with Gradient Checkpointing)"]
        Neck["Shared FPN Neck"]
    end

    subgraph "Detection Heads"
        RPN["RPN Head"]
        ROI["RoI Head (Faster R-CNN)"]
    end

    subgraph "Optimization Goals"
        SupervisedLoss["Supervised Detection Loss (on RGB)"]
        ConsistencyLoss["Cross-Modality Consistency Loss (MSE)"]
    end

    %% Routing
    RGB --> Backbone
    HSI --> Projector
    Projector --> Upsample
    Upsample --> Backbone
    
    Backbone --> Neck
    
    %% RGB Path for Detection
    Neck -- "RGB Features" --> RPN
    RPN --> ROI
    ROI --> SupervisedLoss

    %% Consistency Path
    Neck -- "RGB Features" --> ConsistencyLoss
    Neck -- "HSI Features" --> ConsistencyLoss
```

## 2. Component Roles

### HSI Projector & Upsampler
*   **Role**: Translates raw hyperspectral bands into a 3-channel "latent RGB" representation and matches the spatial resolution of the RGB domain.
*   **Initial State**: Initialized with spectral band-averaging weights (R=53, G=32, B=11).
*   **Target**: Optimized to produce features that "fool" the shared backbone into thinking it's seeing RGB.

### Shared Backbone & Neck
*   **Role**: These layers are the soul of the detector. By sharing them, we ensure that both modalities are mapped into the **exact same feature space**.
*   **Optimization**: They are updated by both the detection loss (to remain accurate) and the consistency loss (to remain modality-agnostic).

### Cross-Modality Consistency (CMC)
*   **Formula**: `loss_consistency = MSE(features_rgb, features_hsi)`
*   **Intuition**: Since the RGB and HSI images show the same ship at the same location, their features at the end of the FPN neck must be identical. This loss "pulls" the HSI representation toward the known-good RGB representation.

## 3. Training vs. Inference

| Feature | Training Phase | Inference (Testing) Phase |
| :--- | :--- | :--- |
| **Inputs** | RGB + HSI (Co-registered) | HSI Only |
| **Backbone** | Bi-directionally updated | Fixed (Inference mode) |
| **Losses** | Detection + Consistency | None (Output BBoxes) |
| **Purpose** | Aligning Modalities | Detecting Objects in HSI |
