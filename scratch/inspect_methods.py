import torch
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.engine.trainer import BaseTrainer
import inspect

print("BaseTrainer methods:")
for name, val in inspect.getmembers(BaseTrainer, predicate=inspect.isfunction):
    print(f"  {name}{inspect.signature(val)}")
