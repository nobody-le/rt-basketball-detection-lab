"""Task 01: replay rt.mp4 frames 200 and 250 and save their detections."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VIDEO = ROOT / "data-src" / "rt.mp4"
WEIGHTS = ROOT / "yolo11n.pt"
FRAMES = (200, 250)
IMGSZ = 640
CONF = 0.10
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task01/replay_frames.py"

for path in (VIDEO, WEIGHTS):
    if not path.is_file():
        raise SystemExit(f"缺少 {path}，先在仓库根目录运行 python lab.py prepare")

reader = cv2.VideoCapture(str(VIDEO))
if not reader.isOpened():
    raise SystemExit(f"无法打开 {VIDEO}")
fps = reader.get(cv2.CAP_PROP_FPS)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames: dict[int, np.ndarray] = {}
index = 0
try:
    while index <= max(FRAMES):
        ok, frame = reader.read()
        if not ok:
            raise SystemExit(f"无法解码第 {index} 帧")
        if index in FRAMES:
            frames[index] = frame.copy()
        index += 1
finally:
    reader.release()

model = YOLO(str(WEIGHTS))
names = {name.lower(): i for i, name in model.names.items()}
target = names["sports ball"]

records = []
for number in FRAMES:
    result = model.predict(
        frames[number], classes=[target], imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False,
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
    records.append({
        "frame": number,
        "time_s": round(number / fps, 4),
        "width": width,
        "height": height,
        "imgsz": IMGSZ,
        "conf": CONF,
        "classes": ["sports ball"],
        "detections": detections,
        "center_xy": center,
    })

jsonl = OUT / "replay-200-250.jsonl"
jsonl.write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)

fields = ["frame", "time_s", "width", "height", "detections", "top_confidence",
          "bbox_xyxy", "center_x", "center_y"]
with (OUT / "replay-200-250.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        top = max(record["detections"], key=lambda det: det["confidence"]) if record["detections"] else None
        writer.writerow({
            "frame": record["frame"],
            "time_s": record["time_s"],
            "width": record["width"],
            "height": record["height"],
            "detections": len(record["detections"]),
            "top_confidence": top["confidence"] if top else "",
            "bbox_xyxy": " ".join(str(v) for v in top["bbox_xyxy"]) if top else "",
            "center_x": record["center_xy"]["x"] if record["center_xy"] else "",
            "center_y": record["center_xy"]["y"] if record["center_xy"] else "",
        })


def annotate(frame, record):
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, 46), (32, 32, 32), -1)
    cv2.putText(panel, f"frame {record['frame']}  t={record['time_s']:.3f}s", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not record["detections"]:
        cv2.putText(panel, "no detection", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        return panel
    for det in record["detections"]:
        x1, y1, x2, y2 = (int(round(value)) for value in det["bbox_xyxy"])
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(panel, f"{det['class']} {det['confidence']:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    if record["center_xy"]:
        cx = int(round(record["center_xy"]["x"]))
        cy = int(round(record["center_xy"]["y"]))
        cv2.drawMarker(panel, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(panel, f"({record['center_xy']['x']:.1f}, {record['center_xy']['y']:.1f})",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return panel


composite = np.hstack([annotate(frames[number], record) for number, record in zip(FRAMES, records)])
if not cv2.imwrite(str(OUT / "replay-200-250.png"), composite):
    raise SystemExit("无法写入 replay-200-250.png")

for record in records:
    print(json.dumps(record, ensure_ascii=False))
print(f"fps={fps:.4f} command={COMMAND}")
