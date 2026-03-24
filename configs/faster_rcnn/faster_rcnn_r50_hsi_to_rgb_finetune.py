_base_ = [
    './faster_rcnn_r50_hsi_to_rgb.py'
]

# Ensure we import the custom freeze hook
custom_imports = dict(
    imports=['tools_new.freeze_hook'],
    allow_failed_imports=False
)

# Load the projected RGB checkpoint
load_from = 'work_dirs/faster_rcnn_r50_rgb/rgb_projected.pth'

# Optimizer Configuration (Reduce learning rate for fine-tuning)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# Learning Rate schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# Register the Freeze Hook (REMOVED FOR FULL FINE-TUNING)
# custom_hooks = [
#     dict(type='FreezeExceptProjectorHook')
# ]

# Set the work directory internally 
work_dir = './work_dirs/faster_rcnn_r50_hsi_to_rgb_finetune'
