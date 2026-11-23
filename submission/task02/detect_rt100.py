"""Task 02: run yolo11n on frame rt-100 and save the boxed result."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMAGE = ROOT / "dataset-curated" / "images" / "val" / "rt-100.jpg"
WEIGHTS = ROOT / "yolo11n.pt"
IMGSZ = 640
CONF = 0.25
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task02/detect_rt100.py"

if not IMAGE.is_file():
    raise SystemExit(f"缺少 {IMAGE.relative_to(ROOT)}，先在仓库根目录运行 python lab.py prepare")
if not WEIGHTS.is_file():
    raise SystemExit(f"缺少 {WEIGHTS.name}，先运行 python lab.py prepare")

image = cv2.imread(str(IMAGE))
if image is None:
    raise SystemExit(f"无法读取 {IMAGE}")

model = YOLO(str(WEIGHTS))
names = {name.lower(): index for index, name in model.names.items()}
target = names["sports ball"]

start = time.perf_counter()
result = model.predict(
    image, classes=[target], imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False,
)[0]
elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

detections = [
    {
        "bbox_xyxy": [round(value, 2) for value in box],
        "class": model.names[target],
        "class_id": target,
        "confidence": round(float(score), 4),
    }
    for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
]

annotated = image.copy()
for det in detections:
    x1, y1, x2, y2 = (int(round(value)) for value in det["bbox_xyxy"])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"{det['class']} {det['confidence']:.2f}"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(0, y1 - th - base - 4)
    cv2.rectangle(annotated, (x1, top), (x1 + tw + 4, top + th + base + 4), (0, 0, 255), -1)
    cv2.putText(annotated, label, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

png = OUT / "rt-100.png"
if not cv2.imwrite(str(png), annotated):
    raise SystemExit(f"无法写入 {png}")

record = {
    "task": "task02",
    "image": str(IMAGE.relative_to(ROOT)),
    "image_size": {"width": image.shape[1], "height": image.shape[0]},
    "weights": WEIGHTS.name,
    "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
    "device": "cpu",
    "imgsz": IMGSZ,
    "conf": CONF,
    "classes": ["sports ball"],
    "detections": detections,
    "elapsed_ms": elapsed_ms,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
    },
    "result_image": str(png.relative_to(ROOT)),
    "command": COMMAND,
}

json_path = OUT / "result.json"
json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
