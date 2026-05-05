import json
import os

path = "data/train_coco/annotations.json"
if os.path.exists(path):
    with open(path, 'r') as f:
        data = json.load(f)
        print("Keys:", data.keys())
        if 'annotations' in data and len(data['annotations']) > 0:
            print("First annotation sample:")
            print(json.dumps(data['annotations'][0], indent=2))
        else:
            print("No annotations found.")
else:
    print(f"File {path} not found.")
