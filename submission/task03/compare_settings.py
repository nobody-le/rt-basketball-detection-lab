"""Task 03: compare three inference settings on the 14 validation frames."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab import DATA, MATCH_IOU, WEIGHTS, group, iou, load_detector, predict_frame, read_annotations  # noqa: E402

OUT = Path(__file__).resolve().parent
CONFIGS = (
    {"id": "A", "imgsz": 640, "conf": 0.10},
    {"id": "B", "imgsz": 320, "conf": 0.10},
    {"id": "C", "imgsz": 320, "conf": 0.25},
)
SHOW_FRAMES = (25, 275)
CHECK_FRAMES = (25, 145, 150, 180, 200, 275)
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task03/compare_settings.py"

rows = [row for row in read_annotations() if row["split"] == "val"]
model, target = load_detector(WEIGHTS)

records = []
stats = []
for config in CONFIGS:
    ball_total = edge_total = empty_total = 0
    ball_hits = edge_hits = false_alarms = 0
    for row in rows:
        image_path = DATA / "images" / "val" / f"{row['stem']}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"缺少 {image_path}，先运行 python lab.py prepare")
        detections, elapsed = predict_frame(
            model, target, image, config["imgsz"], config["conf"],
        )
        label = row["box"]
        best_iou = max((iou(label, det["bbox_xyxy"]) for det in detections), default=0) if label else 0
        hit = bool(label) and best_iou >= MATCH_IOU
        category = group(row)
        if label:
            ball_total += 1
            ball_hits += hit
            if category == "edge":
                edge_total += 1
                edge_hits += hit
        else:
            empty_total += 1
            false_alarms += bool(detections)
        records.append({
            "config": config["id"],
            "imgsz": config["imgsz"],
            "conf": config["conf"],
            "frame": row["frame"],
            "stem": row["stem"],
            "group": category,
            "annotation_xyxy": label,
            "detections": detections,
            "best_iou": round(best_iou, 4),
            "matched": hit,
            "false_alarm": bool(not label and detections),
            "elapsed_ms": elapsed,
        })
    stats.append({
        **config,
        "ball_matched": ball_hits,
        "ball_total": ball_total,
        "edge_matched": edge_hits,
        "edge_total": edge_total,
        "empty_false_alarm": false_alarms,
        "empty_total": empty_total,
    })

(OUT / "results.jsonl").write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)
(OUT / "stats.json").write_text(
    json.dumps({
        "task": "task03",
        "weights": WEIGHTS.name,
        "match_iou": MATCH_IOU,
        "split": "val",
        "frames": len(rows),
        "configs": stats,
        "command": COMMAND,
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

with (OUT / "stats.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["config", "imgsz", "conf", "ball_matched", "ball_total",
                     "edge_matched", "edge_total", "false_alarm", "empty_total"])
    for stat in stats:
        writer.writerow([stat["id"], stat["imgsz"], stat["conf"], stat["ball_matched"],
                         stat["ball_total"], stat["edge_matched"], stat["edge_total"],
                         stat["empty_false_alarm"], stat["empty_total"]])

with (OUT / "frame-checks.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame", "group", "config", "imgsz", "conf", "detections",
                     "top_confidence", "best_iou", "matched", "false_alarm"])
    for record in records:
        if record["frame"] not in CHECK_FRAMES:
            continue
        top = max((det["confidence"] for det in record["detections"]), default="")
        writer.writerow([record["frame"], record["group"], record["config"], record["imgsz"],
                         record["conf"], len(record["detections"]), top,
                         record["best_iou"], record["matched"], record["false_alarm"]])


def annotate(image, record):
    panel = image.copy()
    panel = cv2.copyMakeBorder(panel, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    height, width = panel.shape[:2]
    header = 46
    canvas = cv2.copyMakeBorder(panel, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    cv2.putText(canvas, f"{record['config']}  imgsz={record['imgsz']}  conf={record['conf']:.2f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    verdict = f"match IoU={record['best_iou']:.2f}" if record["annotation_xyxy"] else "no ball"
    if record["annotation_xyxy"] and not record["matched"]:
        verdict = f"miss (IoU={record['best_iou']:.2f})"
    cv2.putText(canvas, verdict, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 255, 120) if record["matched"] else (80, 160, 255), 1, cv2.LINE_AA)
    if record["annotation_xyxy"]:
        x1, y1, x2, y2 = (int(round(v)) for v in record["annotation_xyxy"])
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), (0, 200, 0), 1)
    for det in record["detections"]:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), (0, 0, 255), 2)
        cv2.putText(canvas, f"{det['confidence']:.2f}", (x1 + 2, y1 + header - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


for frame in SHOW_FRAMES:
    panels = []
    for config in CONFIGS:
        record = next(r for r in records if r["frame"] == frame and r["config"] == config["id"])
        image = cv2.imread(str(DATA / "images" / "val" / f"rt-{frame:03d}.jpg"))
        panels.append(annotate(image, record))
    comparison = cv2.hconcat(panels)
    path = OUT / f"frame-{frame:03d}-compare.png"
    if not cv2.imwrite(str(path), comparison):
        raise SystemExit(f"无法写入 {path}")

print(json.dumps({"stats": stats, "outputs": [f"frame-{f:03d}-compare.png" for f in SHOW_FRAMES]},
                 ensure_ascii=False, indent=2))
