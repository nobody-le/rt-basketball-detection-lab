"""Task 06: summarize PyTorch vs ONNX Runtime and check switch conditions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
COORD_TOL = 0.01
CONF_TOL = 1e-4
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task06/summarize.py"

comparison = json.loads((OUT / "backend-comparison.json").read_text(encoding="utf-8"))
frames = []
for frame in comparison["frames"]:
    pt = frame["pytorch"]
    onnx = frame["onnxruntime"]
    counts_match = len(pt) == len(onnx)
    max_coord = 0.0
    max_conf = 0.0
    classes_match = counts_match
    if counts_match:
        for left, right in zip(pt, onnx):
            max_coord = max(max_coord, max(abs(a - b) for a, b in zip(left["bbox_xyxy"], right["bbox_xyxy"])))
            max_conf = max(max_conf, abs(left["confidence"] - right["confidence"]))
            classes_match = classes_match and left["class"] == right["class"]
    boxes_match = counts_match and max_coord <= COORD_TOL
    confidence_match = counts_match and max_conf <= CONF_TOL
    empty = (len(pt) == 0) == (len(onnx) == 0)
    frames.append({
        "frame": frame["frame"],
        "pytorch_count": len(pt),
        "pytorch_box": pt[0]["bbox_xyxy"] if pt else None,
        "pytorch_class": pt[0]["class"] if pt else None,
        "pytorch_conf": pt[0]["confidence"] if pt else None,
        "onnx_count": len(onnx),
        "onnx_box": onnx[0]["bbox_xyxy"] if onnx else None,
        "onnx_class": onnx[0]["class"] if onnx else None,
        "onnx_conf": onnx[0]["confidence"] if onnx else None,
        "max_coord_delta": max_coord,
        "max_conf_delta": max_conf,
        "counts_match": counts_match,
        "boxes_match": boxes_match,
        "classes_match": classes_match,
        "confidence_match": confidence_match,
        "empty_match": empty,
    })

input_ok = (
    comparison["pytorch_input_shapes"] == [[1, 3, 416, 416]] * len(frames)
    and comparison["onnx_input_shape"] == [1, 3, 416, 416]
)
conditions = {
    "input_size_416": input_ok,
    "counts_match": all(item["counts_match"] for item in frames),
    "boxes_match": all(item["boxes_match"] for item in frames),
    "classes_match": all(item["classes_match"] for item in frames),
    "confidence_match": all(item["confidence_match"] for item in frames),
    "empty_result_match": all(item["empty_match"] for item in frames),
}

result = {
    "task": "task06",
    "preprocessing": {"imgsz": comparison["imgsz"], "rect": comparison["rect"],
                      "conf": comparison["conf"], "match_note": "boxes in original-image pixels"},
    "inputs": {
        "pytorch": comparison["pytorch_input_shapes"],
        "onnxruntime": comparison["onnx_input_shape"],
    },
    "tolerances": {"coord_px": COORD_TOL, "confidence": CONF_TOL},
    "conditions": conditions,
    "switch_ready": all(conditions.values()),
    "frames": frames,
    "command": COMMAND,
}
(OUT / "switch-decision.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)

with (OUT / "backend-table.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["frame", "pytorch_count", "pytorch_box", "pytorch_class", "pytorch_conf",
                     "onnx_count", "onnx_box", "onnx_class", "onnx_conf",
                     "max_coord_delta", "max_conf_delta", "boxes_match", "classes_match",
                     "confidence_match", "empty_match"])
    for item in frames:
        writer.writerow([
            item["frame"], item["pytorch_count"],
            " ".join(f"{v:.4f}" for v in item["pytorch_box"]) if item["pytorch_box"] else "",
            item["pytorch_class"] or "",
            "" if item["pytorch_conf"] is None else f"{item['pytorch_conf']:.8f}",
            item["onnx_count"],
            " ".join(f"{v:.4f}" for v in item["onnx_box"]) if item["onnx_box"] else "",
            item["onnx_class"] or "",
            "" if item["onnx_conf"] is None else f"{item['onnx_conf']:.8f}",
            f"{item['max_coord_delta']:.8f}", f"{item['max_conf_delta']:.10f}",
            item["boxes_match"], item["classes_match"], item["confidence_match"], item["empty_match"],
        ])

print(json.dumps(result, ensure_ascii=False, indent=2))
