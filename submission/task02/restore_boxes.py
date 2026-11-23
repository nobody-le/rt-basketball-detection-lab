"""Task 02: restore boxes from the CVAT Ultralytics YOLO export onto originals."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASET = OUT / "yolo"
NAMES = ("rt-150", "plos02-055")
COMMAND = "env -u PYTHONPATH .venv/bin/python submission/task02/restore_boxes.py"

records = []
for stem in NAMES:
    image_path = DATASET / "images" / "train" / f"{stem}.jpg"
    label_path = DATASET / "labels" / "train" / f"{stem}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise SystemExit(f"缺少导出文件：{image_path} 或 {label_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"无法读取 {image_path}")
    height, width = image.shape[:2]

    lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f"{label_path.name} 期望 1 个框，实际 {len(lines)}")
    class_id, cx, cy, bw, bh = lines[0]
    class_id, cx, cy, bw, bh = int(class_id), float(cx), float(cy), float(bw), float(bh)
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    pixel_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

    annotated = image.copy()
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    thickness = 2
    cv2.rectangle(annotated, p1, p2, (0, 0, 255), thickness)
    label = f"basketball {pixel_box[0]:.1f},{pixel_box[1]:.1f} - {pixel_box[2]:.1f},{pixel_box[3]:.1f}"
    font_scale = 0.5 if width <= 640 else 0.8
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    ty = max(th + base + 4, p1[1])
    cv2.rectangle(annotated, (p1[0], ty - th - base - 4), (p1[0] + tw + 4, ty), (0, 0, 255), -1)
    cv2.putText(annotated, label, (p1[0] + 2, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    output_path = OUT / f"{stem}-box.png"
    if not cv2.imwrite(str(output_path), annotated):
        raise SystemExit(f"无法写入 {output_path}")

    records.append({
        "image": str(image_path.relative_to(ROOT)),
        "size": {"width": width, "height": height},
        "label_file": str(label_path.relative_to(ROOT)),
        "yolo_line": " ".join(lines[0]),
        "class_id": class_id,
        "class_name": "basketball",
        "normalized_cxcywh": [cx, cy, bw, bh],
        "pixel_xyxy": pixel_box,
        "output_image": str(output_path.relative_to(ROOT)),
    })

result = {
    "task": "task02",
    "format": "Ultralytics YOLO Detection 1.0",
    "export": "submission/task02/cvat-export.zip",
    "restored": records,
    "command": COMMAND,
}
json_path = OUT / "restored-boxes.json"
json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
