from mmdet.datasets import build_dataset
from mmcv import Config

cfg = Config.fromfile('configs/faster_rcnn/fold_1_vitae_s_3x_hbb.py')
# Explicitly set filter_empty_gt to False to see all images
cfg.data.train.filter_empty_gt = False
dataset = build_dataset(cfg.data.train)
print(f"Dataset length (filter_empty_gt=False): {len(dataset)}")
print(f"Dataset CLASSES: {dataset.CLASSES}")

# Now check how many have annotations
with_ann = 0
for i in range(len(dataset)):
    ann = dataset.get_ann_info(i)
    if len(ann['bboxes']) > 0:
        with_ann += 1
print(f"Samples with annotations: {with_ann}")

# Check one annotation directly
if len(dataset) > 0:
    print(f"Sample 0 raw ann: {dataset.get_ann_info(0)}")
