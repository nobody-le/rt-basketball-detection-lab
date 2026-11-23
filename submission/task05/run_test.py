"""Task 05: independent Jackson test at 416 input and 0.10 confidence."""

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
IMGSZ = 416
CONF = 0.10
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task05/run_test.py"

rows = [row for row in read_annotations() if row["split"] == "test"]
model, target = load_detector(WEIGHTS)

records = []
review_frames = []
for row in rows:
    image = cv2.imread(str(DATA / "images" / "test" / f"{row['stem']}.jpg"))
    if image is None:
        raise SystemExit(f"缺少 test 图 {row['stem']}.jpg，先运行 python lab.py prepare")
    detections, elapsed = predict_frame(model, target, image, IMGSZ, CONF)
    label = row["box"]
    best_iou = max((iou(label, det["bbox_xyxy"]) for det in detections), default=0) if label else 0
    matched = bool(label) and best_iou >= MATCH_IOU

    if label and not matched and detections:
        reason = "unmatched_extra_box"
    elif label and not matched:
        reason = "miss"
    elif not label and detections:
        reason = "false_alarm"
    elif matched and len(detections) > 1:
        reason = "extra_box"
    else:
        reason = "ok"
    review = reason != "ok"
    if review:
        review_frames.append(row["frame"])

    records.append({
        "frame": row["frame"],
        "image": f"{row['stem']}.jpg",
        "group": group(row),
        "annotation_xyxy": label,
        "detections": detections,
        "top_confidence": max((det["confidence"] for det in detections), default=None),
        "best_iou": round(best_iou, 4),
        "matched": matched,
        "review": review,
        "reason": reason,
        "elapsed_ms": elapsed,
    })

(OUT / "jackson-test.jsonl").write_text(
    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    encoding="utf-8",
)

with (OUT / "test-frames.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame", "image", "group", "gt_xyxy", "detections",
                     "top_confidence", "best_iou", "matched", "review", "reason"])
    for record in records:
        writer.writerow([
            record["frame"],
            record["image"],
            record["group"],
            " ".join(str(v) for v in record["annotation_xyxy"]) if record["annotation_xyxy"] else "",
            len(record["detections"]),
            "" if record["top_confidence"] is None else record["top_confidence"],
            record["best_iou"],
            record["matched"],
            record["review"],
            record["reason"],
        ])

ball_frames = sum(1 for record in records if record["annotation_xyxy"])
matched = sum(1 for record in records if record["matched"])
empty_frames = len(records) - ball_frames
false_alarms = sum(1 for record in records if not record["annotation_xyxy"] and record["detections"])
extra_boxes = sum(len(record["detections"]) for record in records) - matched
summary = {
    "task": "task05",
    "weights": WEIGHTS.name,
    "device": "cpu",
    "imgsz": IMGSZ,
    "conf": CONF,
    "match_iou": MATCH_IOU,
    "split": "test (Jackson only)",
    "frames": len(records),
    "ball_frames": ball_frames,
    "matched": matched,
    "missed": ball_frames - matched,
    "empty_frames": empty_frames,
    "false_alarm_frames": false_alarms,
    "extra_boxes": extra_boxes,
    "review_frames": review_frames,
    "ok_frames": [record["frame"] for record in records if not record["review"]],
    "command": COMMAND,
}
(OUT / "test-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)

with (OUT / "grouping.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["source", "frames", "correct_split", "reported_wrong_split", "note"])
    for source in ("plos01", "plos02", "plos03", "plos06", "plos07", "plos09"):
        count = sum(1 for row in read_annotations() if row["source"] == source)
        writer.writerow([source, count, "train", "train", ""])
    writer.writerow(["rt", 14, "val", "train: 1; val: 13", "rt-125 在 train"])
    writer.writerow(["jackson", 8, "test", "val: 1; test: 7", "jackson-011 在 val"])

print(json.dumps(summary, ensure_ascii=False, indent=2))
