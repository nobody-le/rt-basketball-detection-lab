"""Task 05: draw GT and predicted boxes for the 8 Jackson test frames."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMAGES = ROOT / "dataset-curated" / "images" / "test"
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task05/draw_comparison.py"

records = [json.loads(line) for line in (OUT / "jackson-test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def annotate(record, width=None):
    image = cv2.imread(str(IMAGES / record["image"]))
    if image is None:
        raise SystemExit(f"缺少 {IMAGES / record['image']}，先运行 python lab.py prepare")
    height = image.shape[0]
    if width is not None:
        height = round(height * width / image.shape[1])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scale = image.shape[1] / 1440
    header = max(36, round(70 * scale))
    font = max(0.4, 0.8 * scale)
    thickness = max(1, round(3 * scale))
    canvas = cv2.copyMakeBorder(image, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))

    if record["annotation_xyxy"]:
        verdict = f"match IoU={record['best_iou']:.4f}" if record["matched"] else f"miss IoU={record['best_iou']:.4f}"
    else:
        verdict = "GT: no ball"
    if not record["detections"]:
        verdict += " | no prediction"
    cv2.putText(canvas, f"frame {record['frame']:03d}  {verdict}", (6, round(header * 0.62)),
                cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)
    legend = "GT green / prediction red" if record["annotation_xyxy"] else "no GT box"
    cv2.putText(canvas, legend, (6, header - max(3, round(6 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, (190, 190, 190), max(1, round(1 * scale)), cv2.LINE_AA)

    if record["annotation_xyxy"]:
        x1, y1, x2, y2 = (round(v * scale) for v in record["annotation_xyxy"])
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), (0, 200, 0), thickness)
    for det in record["detections"]:
        x1, y1, x2, y2 = (round(v * scale) for v in det["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), (0, 0, 255), thickness)
        cv2.putText(canvas, f"pred {det['confidence']:.2f}", (x1 + 2, max(header + round(14 * scale), y1 + header - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font * 0.85, (0, 0, 255), max(1, round(1 * scale)), cv2.LINE_AA)
    return canvas


index = []
for record in records:
    output = OUT / f"jackson-{record['frame']:03d}-compare.jpg"
    image = annotate(record, width=None)
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise SystemExit(f"无法写入 {output}")
    index.append({
        "frame": record["frame"],
        "output": str(output.relative_to(ROOT)),
        "gt_xyxy": record["annotation_xyxy"],
        "pred_boxes": [det["bbox_xyxy"] for det in record["detections"]],
        "pred_confidences": [det["confidence"] for det in record["detections"]],
        "best_iou": record["best_iou"],
        "matched": record["matched"],
        "review": record["review"],
        "reason": record["reason"],
    })

panels = [annotate(record, width=360) for record in records]
rows = [np.hstack(panels[:4]), np.hstack(panels[4:])]
grid = np.vstack(rows)
if not cv2.imwrite(str(OUT / "jackson-8-grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92]):
    raise SystemExit("无法写入 jackson-8-grid.jpg")

with (OUT / "comparison-index.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame", "output", "gt_xyxy", "pred_boxes", "pred_confidences",
                     "best_iou", "matched", "review", "reason"])
    for item in index:
        writer.writerow([
            item["frame"], item["output"],
            " ".join(str(v) for v in item["gt_xyxy"]) if item["gt_xyxy"] else "",
            " | ".join(" ".join(str(v) for v in box) for box in item["pred_boxes"]),
            " ".join(f"{value:.4f}" for value in item["pred_confidences"]),
            item["best_iou"], item["matched"], item["review"], item["reason"],
        ])

print(json.dumps({"images": len(index), "grid": "submission/task05/jackson-8-grid.jpg",
                  "index": "submission/task05/comparison-index.csv", "command": COMMAND},
                 ensure_ascii=False, indent=2))
