"""Task 06: draw PyTorch vs ONNX Runtime detections for rt-100/150/200."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMAGES = ROOT / "dataset-curated" / "images" / "val"
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task06/draw_compare.py"

comparison = json.loads((OUT / "backend-comparison.json").read_text(encoding="utf-8"))


def annotate(image, detections, backend):
    canvas = cv2.copyMakeBorder(image, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    frame = detections["frame"]
    boxes = detections["detections"]
    cv2.putText(canvas, f"frame {frame:03d} | {backend}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if boxes:
        top = boxes[0]
        cv2.putText(canvas, f"{top['class']}  conf={top['confidence']:.6f}", (6, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1, cv2.LINE_AA)
        for det in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
            cv2.rectangle(canvas, (x1, y1 + 52), (x2, y2 + 52), (0, 0, 255), 2)
            cv2.putText(canvas, f"{det['confidence']:.3f}", (x1 + 2, y1 + 52 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "no detection (empty)", (6, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 255), 1, cv2.LINE_AA)
    return canvas


for item in comparison["frames"]:
    frame = item["frame"]
    image = cv2.imread(str(IMAGES / f"rt-{frame:03d}.jpg"))
    if image is None:
        raise SystemExit(f"缺少 rt-{frame:03d}.jpg，先运行 python lab.py prepare")
    left = annotate(image, {"frame": frame, "detections": item["pytorch"]}, "PyTorch")
    right = annotate(image, {"frame": frame, "detections": item["onnxruntime"]}, "ONNX Runtime")
    output = OUT / f"rt-{frame:03d}-backend-compare.png"
    if not cv2.imwrite(str(output), np.hstack([left, right])):
        raise SystemExit(f"无法写入 {output}")

print(json.dumps({"images": [f"rt-{item['frame']:03d}-backend-compare.png" for item in comparison["frames"]],
                  "command": COMMAND}, ensure_ascii=False, indent=2))
