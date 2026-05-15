_base_ = './fold_3_swin_t_3x_hbb.py'

# We update the normalization to match the dark HSI dataset
# HSI Mean (RGB): [26.4, 37.5, 17.2]
# HSI Std (RGB): [13.9, 20.6, 21.5]
img_norm_cfg = dict(
    mean=[26.4, 37.5, 17.2], 
    std=[13.9, 20.6, 21.5], 
    to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg), # Use HSI normalization
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    test=dict(
        pipeline=test_pipeline
    )
)
