# 任务二 · CVAT 标注与 YOLO Detection 导出

在 CVAT 2.83.0 中为 4 张图标注 `basketball`，导出 `Ultralytics YOLO Detection 1.0` 数据集，再用导出坐标在 `rt-150`、`plos02-055` 原图上还原边界框。

标注框为在 CVAT 中人工核对的 `basketball` 框（`source=manual`），由 CVAT 导出；导出文件是本任务的提交物。

## CVAT 任务

| 项 | 值 |
|---|---|
| CVAT | 2.83.0，`http://localhost:8080` |
| 项目 | `task02-basketball`（id 1） |
| 标签 | `basketball`（id 1） |
| 任务 | `task02-4frames`（id 1，4 帧） |
| 导出格式 | `Ultralytics YOLO Detection 1.0`（`save_images=true`） |

| 帧 | 文件 | CVAT 框 `xyxy` | 导出标签 |
|---:|---|---|---|
| 0 | `plos02-055.jpg` | `[1086.52, 575.7, 1155.52, 644.7]` | `0 0.583865 0.565000 0.035937 0.063889` |
| 1 | `rt-125.jpg` | `[45.66, 303, 164.66, 425]` | `0 0.292111 0.568750 0.330556 0.190625` |
| 2 | `rt-150.jpg` | `[0, 241.6, 35.7, 348.7]` | `0 0.049583 0.461172 0.099167 0.167344` |
| 3 | `rt-200.jpg` | 无球，空标注 | 无标签文件 |

## 提交的导出

- `CVAT_2.zip`：人工标注后的 `CVAT for images 1.1` 导出（含 `annotations.xml` 与 4 张图像），用于逐框核对
- `cvat-export.zip`：按同一任务标注重新导出（`save_images=true`，含图像），作为本任务提交包
- `yolo/`：`cvat-export.zip` 解包后的内容

```text
yolo/
├── data.yaml
├── train.txt
├── images/train/{plos02-055,rt-125,rt-150,rt-200}.jpg
└── labels/train/{plos02-055,rt-125,rt-150}.txt
```

`rt-200.jpg` 无目标，因此 `labels/train/` 中没有对应 `.txt`，这是 YOLO 的空帧表示。

## 用导出坐标还原边界框

```bash
env -u PYTHONPATH .venv/bin/python submission/task02/restore_boxes.py
```

脚本读取 `yolo/labels/train/*.txt` 的归一化 `cx cy w h`，按原图宽高换算回像素；同时读取 `CVAT_2.zip` 的 `annotations.xml` 逐框核对，本次两帧的 CVAT 框与 YOLO 标签换算结果一致：

| 图像 | 尺寸 | 归一化 `cx cy w h` | 还原像素框 `xyxy` |
|---|---|---|---|
| `rt-150.jpg` | 360×640 | `0.049583 0.461172 0.099167 0.167344` | `[0, 241.6, 35.7, 348.7]` |
| `plos02-055.jpg` | 1920×1080 | `0.583865 0.565000 0.035937 0.063889` | `[1086.52, 575.7, 1155.52, 644.7]` |

输出：

- `rt-150-box.png`
- `plos02-055-box.png`
- `restored-boxes.json`（归一化坐标、像素框、复跑命令）

## 重新导出

CVAT 2.83 的导出分两步：`POST /api/tasks/1/dataset/export?format=Ultralytics%20YOLO%20Detection%201.0&save_images=true` 取得 `rq_id`，轮询 `GET /api/requests/{rq_id}`，完成后下载 `result_url`。解包可用：

```bash
python3 -m zipfile -e submission/task02/cvat-export.zip submission/task02/yolo
```

CVAT 项目、任务、帧映射和标注记录见 `cvat-task.json`。
