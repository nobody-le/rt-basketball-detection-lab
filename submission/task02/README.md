# 任务二 · rt-100 单帧检测

用 `yolo11n.pt` 在 CPU 上检测 `dataset-curated/images/val/rt-100.jpg`，只保留 `sports ball` 类，并在原图像素坐标下记录框与置信度。

## 复跑入口

先在仓库根目录准备素材（已准备可跳过）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python lab.py prepare
```

再运行本任务的检测并重写结果：

```bash
env -u PYTHONPATH .venv/bin/python submission/task02/detect_rt100.py
```

## 运行记录

| 项 | 值 |
|---|---|
| 权重 | `yolo11n.pt`（SHA-256 `0ebbc80d…644ee1`） |
| 设备 | CPU |
| 输入尺寸 | 640 |
| 置信度阈值 | 0.25 |
| 类别 | 仅 `sports ball`（class 32） |
| 原图 | 360×640 |
| 检测框 `xyxy` | `[139.41, 390.48, 260.47, 516.74]` |
| 置信度 | 0.5129 |
| 耗时 | 84.22 ms |

- `rt-100.png`：带框结果图
- `result.json`：完整运行记录（含权重校验值、依赖版本、复跑命令）

> `lab.py prepare` 会生成 `dataset-curated/`；该目录与权重被 `.gitignore` 忽略，因此复跑前必须执行一次。
