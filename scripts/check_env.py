"""
check_env.py — Phase 0 Step 1: environment verification
Run: python check_env.py
"""

import platform
import sys

print("=" * 60)
print("Python   :", sys.version.split()[0])
print("Platform :", platform.platform())
print("=" * 60)

import torch

print("torch          :", torch.__version__)
print("cuda build     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
print("device         :", torch.cuda.get_device_name(0))
print("capability     :", torch.cuda.get_device_capability(0))
print("arch_list      :", torch.cuda.get_arch_list())

# The real test: is_available() can return True while kernels are missing.
x = torch.randn(1000, 1000, device="cuda")
print("gpu matmul     :", (x @ x).sum().item())

props = torch.cuda.get_device_properties(0)
print("total VRAM     : %.1f GB" % (props.total_memory / 1024**3))
print("=" * 60)

import ultralytics
from ultralytics import YOLO

print("ultralytics    :", ultralytics.__version__)

for name in ("yolov10n.pt", "yolo26n.pt"):
    m = YOLO(name)
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"{name:14s} : OK ({n_params:,} params)")

print("=" * 60)
print("Phase 0 Step 1: PASS")