"""Task 01: compare rt-225 and its horizontal flip by top-box center."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = OUT / "rt-225.jpg"
FLIPPED = OUT / "rt-225-flip.png"
COMPARE = OUT / "rt-225-compare.png"
WEIGHTS = ROOT / "yolo11n.pt"
IMGSZ = 640
FIRST_CONF = 0.25
FLIP_CONF = 0.10
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task01/compare_flip.py"

for path in (SOURCE, WEIGHTS):
    if not path.is_file():
        raise SystemExit(f"缺少 {path}")

image = cv2.imread(str(SOURCE))
if image is None:
    raise SystemExit(f"无法读取 {SOURCE}")
height, width = image.shape[:2]

model = YOLO(str(WEIGHTS))
names = {name.lower(): index for index, name in model.names.items()}
target = names["sports ball"]


def run(frame, conf):
    result = model.predict(
        frame, classes=[target], imgsz=IMGSZ, conf=conf, device="cpu", verbose=False,
    )[0]
    detections = [
        {
            "bbox_xyxy": [round(value, 2) for value in box],
            "class": model.names[target],
            "confidence": round(float(score), 4),
        }
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
    ]
    top = max(detections, key=lambda det: det["confidence"]) if detections else None
    center = None
    if top:
        x1, y1, x2, y2 = top["bbox_xyxy"]
        center = {"x": round((x1 + x2) / 2, 1), "y": round((y1 + y2) / 2, 1)}
    return detections, top, center


original_detections, original_top, original_center = run(image, FIRST_CONF)
flipped_image = cv2.flip(image, 1)
if not cv2.imwrite(str(FLIPPED), flipped_image):
    raise SystemExit(f"无法写入 {FLIPPED}")
flipped_detections, flipped_top, flipped_center = run(flipped_image, FLIP_CONF)

mirror = None
if original_center and flipped_center:
    mirror = {
        "expected_flipped_x": round(width - original_center["x"], 1),
        "dx": round(flipped_center["x"] - (width - original_center["x"]), 1),
        "dy": round(flipped_center["y"] - original_center["y"], 1),
    }


def annotate(frame, top, center, conf, title):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (width, 46), (32, 32, 32), -1)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"conf={conf:.2f}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    if top is None:
        cv2.putText(out, "no detection", (6, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        return out
    x1, y1, x2, y2 = (int(round(value)) for value in top["bbox_xyxy"])
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"{top['class']} {top['confidence']:.2f}"
    cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 255), 1, cv2.LINE_AA)
    cx, cy = int(round(center["x"])), int(round(center["y"]))
    cv2.drawMarker(out, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(out, f"({center['x']:.1f}, {center['y']:.1f})", (6, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return out


left = annotate(image, original_top, original_center, FIRST_CONF, "rt-225.jpg")
right = annotate(flipped_image, flipped_top, flipped_center, FLIP_CONF, "rt-225-flip.png")
if not cv2.imwrite(str(COMPARE), np.hstack([left, right])):
    raise SystemExit(f"无法写入 {COMPARE}")

record = {
    "task": "task01",
    "check": "rt-225 horizontal flip center comparison",
    "source": {
        "image": str(SOURCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "size": {"width": width, "height": height},
    },
    "device": "cpu",
    "imgsz": IMGSZ,
    "classes": ["sports ball"],
    "original": {
        "conf": FIRST_CONF,
        "detections": original_detections,
        "center_xy": original_center,
    },
    "flipped": {
        "image": str(FLIPPED.relative_to(ROOT)),
        "conf": FLIP_CONF,
        "detections": flipped_detections,
        "center_xy": flipped_center,
    },
    "mirror": mirror,
    "compare_image": str(COMPARE.relative_to(ROOT)),
    "command": COMMAND,
}
json_path = OUT / "rt-225-centers.json"
json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
