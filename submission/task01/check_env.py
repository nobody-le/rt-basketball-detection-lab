"""Task 01: verify the project .venv and CPU-only PyTorch."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

import cv2
import numpy
import torch
import torchvision
import ultralytics

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"

EXPECTED = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "ultralytics": "8.3.232",
    "opencv-python": "4.11.0.86",
    "numpy": "1.26.4",
}

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str) -> None:
    results.append((label, ok, detail))


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<missing>"


print("== System ==")
print(f"host              : {platform.node()}")
print(f"platform          : {platform.platform()}")
print(f"python executable : {sys.executable}")
print(f"python prefix     : {sys.prefix}")
print(f"python version    : {platform.python_version()}")
check("interpreter is project .venv", Path(sys.executable).is_relative_to(VENV),
      str(sys.executable))
check("sys.prefix is project .venv", Path(sys.prefix) == VENV, sys.prefix)

print("\n== Dependencies (course requirements) ==")
print(f"{'package':<16}{'installed':<14}{'required':<12}status")
for name, required in EXPECTED.items():
    installed = version(name)
    # The CPU index publishes torch/torchvision as 2.6.0+cpu / 0.21.0+cpu.
    ok = installed in (required, f"{required}+cpu")
    check(f"{name}=={required}", ok, installed)
    print(f"{name:<16}{installed:<14}{required:<12}{'OK' if ok else 'MISMATCH'}")

print(f"{'cv2 module':<16}{cv2.__version__:<14}{'':<12}{cv2.__file__}")
print(f"{'torch wheel':<16}{torch.__version__:<14}")
print(f"{'torchvision':<16}{torchvision.__version__:<14}")
print(f"{'ultralytics':<16}{ultralytics.__version__:<14}")

print("\n== CPU-only PyTorch ==")
cpu_only = "+cpu" in torch.__version__
check("torch wheel is +cpu", cpu_only, torch.__version__)
check("torch.version.cuda is None", torch.version.cuda is None, str(torch.version.cuda))
check("CUDA unavailable", not torch.cuda.is_available(), str(torch.cuda.is_available()))
check("CUDA device count is 0", torch.cuda.device_count() == 0, str(torch.cuda.device_count()))
check("cuDNN unavailable", not torch.backends.cudnn.is_available(),
      str(torch.backends.cudnn.is_available()))

a = torch.ones(256, 256)
b = torch.ones(256, 256)
c = a @ b
check("tensor op runs on cpu", c.device.type == "cpu", str(c.device))
print(f"torch.__version__       : {torch.__version__}")
print(f"torch.version.cuda      : {torch.version.cuda}")
print(f"torch.cuda.is_available : {torch.cuda.is_available()}")
print(f"torch.cuda.device_count : {torch.cuda.device_count()}")
print(f"torch.backends.cudnn    : {torch.backends.cudnn.is_available()}")
print(f"matmul result device    : {c.device}  shape={tuple(c.shape)}  value={c[0, 0].item():.0f}")
print(f"torch.get_num_threads   : {torch.get_num_threads()}")

failed = [label for label, ok, _ in results if not ok]
print("\n== Result ==")
print(f"{'PASS' if not failed else 'FAIL'}: {len(results) - len(failed)}/{len(results)} checks")
for label, ok, detail in results:
    if not ok:
        print(f"  FAILED: {label} (actual: {detail})")
sys.exit(1 if failed else 0)
