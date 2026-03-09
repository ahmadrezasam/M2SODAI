# SFDA model configuration
_base_ = [
    './faster_rcnn_r50_hsi.py'
]

# Override the backbone specifically for the source-free domain adaptation
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ProjectUpsampleResNet',
        in_channels=30, # Assuming HSI PCA 30 channels
        target_size=(1600, 1600), # Bridge to original RGB scale
        resnet_cfg=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1, # Can freeze the RGB backbone or keep training
            norm_cfg=norm_cfg,
            zero_init_residual=False,
            norm_eval=True, 
            init_cfg=None,
            style='pytorch'
        )
    )
)

# Use the generated pseudo labels on the training data
data = dict(
    train=dict(
        ann_file='data/train_coco/pseudo_labels_HSI.json',
    )
)

# Set lower learning rate since we are fine-tuning on pseudo-labels
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# Ensure load_from points to our pretrained RGB backbone
# We have mapped `backbone.*` to `backbone.resnet.*` via `tools/convert_rgb_to_sfda.py`
load_from = 'rgb_sfda_mapped.pth'
