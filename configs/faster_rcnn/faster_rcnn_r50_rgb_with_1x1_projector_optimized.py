# Optimized config for 1x1 Projector test
# Re-uses the base projector config but fixes the OOM issue by reducing workers
_base_ = './faster_rcnn_r50_rgb_with_1x1_projector.py'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0  # Set to 0 to avoid OOM from multiple HSI copies in memory
)
