import sys
import os
import platform

print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Current Working Directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
print(f"sys.path: {sys.path}")

try:
    import importlib.metadata
    print("importlib.metadata is available")
except ImportError:
    print("importlib.metadata is NOT available")
