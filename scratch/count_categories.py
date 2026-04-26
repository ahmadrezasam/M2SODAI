import json
import os

files = [
    ('Train (Optical)', 'data/train_coco/annotations.json'),
    ('Val (Optical)', 'data/val_coco/annotations.json'),
    ('Test (Optical/HSI)', 'data/test_coco/annotations.json'),
    ('Test (HSI-RGB)', 'data/test_rgb/annotations_RGB.json')
]

for name, path in files:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        cats = {c['id']: c['name'] for c in data['categories']}
        counts = {cats[c_id]: 0 for c_id in cats}
        for ann in data['annotations']:
            counts[cats[ann['category_id']]] += 1
        print(f"{name}: {counts}")
    else:
        print(f"{name}: Not found at {path}")
