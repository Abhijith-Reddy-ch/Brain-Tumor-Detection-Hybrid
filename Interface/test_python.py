print("Start")
import sys
print(sys.executable)
try:
    import torch
    print("Torch imported")
except ImportError:
    print("Torch not found")
print("End")
