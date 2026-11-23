"""Task 04: full and freeze-first-10 fine-tuning, then common 640/0.25 evaluation."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab import evaluate, load_detector  # noqa: E402

OUT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "yolo11n.pt"
DATA = ROOT / "data.yaml"
RUNS = ROOT / "runs" / "task04"
EVAL = OUT / "eval"
EPOCHS = 5

if not WEIGHTS.is_file() or not DATA.is_file():
    raise SystemExit("缺少 yolo11n.pt 或 data.yaml，先运行 python lab.py prepare")

COMMON = {
    "data": str(DATA),
    "epochs": EPOCHS,
    "imgsz": 416,
    "batch": 8,
    "workers": 0,
    "device": "cpu",
    "seed": 42,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "mosaic": 0.0,
    "mixup": 0.0,
    "scale": 0.20,
    "plots": False,
    "project": str(RUNS),
    "exist_ok": True,
    "verbose": False,
}

CONFIGS = (
    {"name": "full", "freeze": None, "label": "全量微调"},
    {"name": "freeze10", "freeze": 10, "label": "冻结前 10 层"},
)


def parameter_counts(model, freeze):
    """Count total, frozen, and trainable parameters for whole-model or layer-0..N-1 freezing."""
    layers = list(model.model.model)
    total = sum(p.numel() for p in model.model.parameters())
    frozen = sum(p.numel() for layer in layers[:freeze] for p in layer.parameters()) if freeze else 0
    return total, frozen, total - frozen


configs = []
for config in CONFIGS:
    model = YOLO(str(WEIGHTS))
    total, frozen, trainable = parameter_counts(model, config["freeze"])
    kwargs = dict(COMMON, name=config["name"])
    if config["freeze"] is not None:
        kwargs["freeze"] = config["freeze"]
    model.train(**kwargs)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练未生成 {best}")

    target_dir = OUT / config["name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("results.csv", "args.yaml"):
        source = save_dir / filename
        if source.is_file():
            shutil.copy2(source, target_dir / filename)
    configs.append({
        "name": config["name"],
        "label": config["label"],
        "freeze": config["freeze"],
        "epochs": EPOCHS,
        "train_args": {key: value for key, value in COMMON.items() if key not in ("project", "exist_ok")},
        "weights": str(best.relative_to(ROOT)),
        "total_params": total,
        "frozen_params": frozen,
        "trainable_params": trainable,
        "results_csv": str((target_dir / "results.csv").relative_to(ROOT)),
        "args_yaml": str((target_dir / "args.yaml").relative_to(ROOT)),
    })

    model, target = load_detector(best)
    eval_args = SimpleNamespace(weights=best, imgsz=640, conf=0.25, split="val",
                                output=EVAL / f"{config['name']}.jsonl")
    evaluate(eval_args, model, target)

(OUT / "configs.json").write_text(
    json.dumps({"task": "task04", "pretrained": WEIGHTS.name, "configs": configs},
               ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

table = []
for config in CONFIGS:
    summary = json.loads((EVAL / f"{config['name']}.summary.json").read_text(encoding="utf-8"))
    groups = summary["groups"]
    ball_groups = {key: value for key, value in groups.items() if key != "empty"}
    edge = groups.get("edge", {"frames": 0, "hits": 0})
    empty = groups["empty"]
    table.append({
        "config": config["name"],
        "freeze": config["freeze"],
        "epochs": EPOCHS,
        "ball_matched": sum(value["hits"] for value in ball_groups.values()),
        "ball_total": sum(value["frames"] for value in ball_groups.values()),
        "edge_matched": edge["hits"],
        "edge_total": edge["frames"],
        "false_alarm": empty["false_alarm_frames"],
        "empty_total": empty["frames"],
        "imgsz": 640,
        "conf": 0.25,
        "match_iou": summary["match_iou"],
    })

with (OUT / "eval-table.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
(OUT / "eval-table.json").write_text(
    json.dumps({"common": {"imgsz": 640, "conf": 0.25, "match_iou": 0.5}, "rows": table},
               ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(table, ensure_ascii=False, indent=2))
