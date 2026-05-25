import inspect
from ultralytics.engine.trainer import BaseTrainer

print("build_optimizer source code:")
print(inspect.getsource(BaseTrainer.build_optimizer))

print("\n_model_train source code:")
print(inspect.getsource(BaseTrainer._model_train))
