import sys
import mmdet
print(f"MMDET PATH: {mmdet.__file__}")
print("SYS PATH:")
for p in sys.path:
    print(p)
