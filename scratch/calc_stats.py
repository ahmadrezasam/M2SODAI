import numpy as np

map50_values = [0.369, 0.357, 0.436]
mean = np.mean(map50_values)
std = np.std(map50_values, ddof=1) # sample standard deviation

print(f"Mean: {mean:.4f}")
print(f"Std: {std:.4f}")
