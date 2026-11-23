# 任务一 · 环境核验与 rt-100 单帧检测

包含两项内容：项目 `.venv` 与 CPU 版 PyTorch 的核验，以及用 `yolo11n.pt` 对 `rt-100` 的单帧检测。

## 1. 环境核验

核验项目 `.venv` 的 Python 路径、课程锁定依赖版本和 PyTorch 的 CPU 状态。

```bash
env -u PYTHONPATH .venv/bin/python submission/task01/check_env.py | tee submission/task01/environment.log
```

`env -u PYTHONPATH` 用于隔离本机 ROS 2 的 `site-packages`，避免其插到 venv 之前。

本次结果：`PASS: 13/13 checks`，退出码 0。

- Python：`/home/nobody-le/projects/rt-basketball-detection-lab/.venv/bin/python`（3.12.3）
- torch `2.6.0+cpu`、torchvision `0.21.0+cpu`、ultralytics `8.3.232`、opencv-python `4.11.0.86`、numpy `1.26.4`
- CPU 核验：`torch.version.cuda=None`、`torch.cuda.is_available()=False`、`device_count=0`、矩阵运算结果在 `cpu`

> PyTorch CPU 索引安装后的包元数据版本带 `+cpu` 本地后缀，脚本按 `2.6.0` / `2.6.0+cpu` 两种写法视为同一锁定版本。

## 2. rt-100 单帧检测

只保留 `sports ball` 类，在原图像素坐标下记录框与置信度。

```bash
env -u PYTHONPATH .venv/bin/python submission/task01/detect_rt100.py
```

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
| 耗时 | 约 74 ms（本次记录 74.19 ms） |

- `rt-100.png`：带框结果图
- `result.json`：完整运行记录（含权重校验值、依赖版本、复跑命令）

## 3. rt-225 与其左右翻转图的中心点对照

对 `rt-225.jpg` 做首次推理（阈值 0.25），再由该图左右翻转生成 `rt-225-flip.png` 并以阈值 0.10 推理。两者都取最高分框的中心，坐标为该图像素坐标。

```bash
env -u PYTHONPATH .venv/bin/python submission/task01/compare_flip.py
```

| 输入 | 阈值 | 最高分 | 中心 `(x, y)` |
|---|---:|---:|---|
| `rt-225.jpg` | 0.25 | 0.2513 | (78.7, 210.8) |
| `rt-225-flip.png` | 0.10 | 0.2809 | (281.9, 211.3) |

镜像校验：宽 360，期望翻转后 `x = 360 − 78.7 = 281.3`，实测 `281.9`，`Δx = +0.6`，`Δy = +0.5`。

- `rt-225.jpg`：仓库生成的验证原图
- `rt-225-flip.png`：由原图左右翻转得到的 PNG
- `rt-225-centers.json`：中心点、框、置信度和复跑命令
- `rt-225-compare.png`：左右中心点对照图

## 4. rt.mp4 第 200、225 帧回放

从 `data-src/rt.mp4` 顺序解码到第 200、225 帧，按 640 输入、阈值 0.10、仅 `sports ball` 推理；检测为空时保留该帧记录。

```bash
env -u PYTHONPATH .venv/bin/python submission/task01/replay_frames.py
```

| 帧 | 时间 (s) | 检测数 | 最高分 | 框 `xyxy` | 中心 `(x, y)` |
|---:|---:|---:|---:|---|---|
| 200 | 3.3325 | 0 | — | — | — |
| 225 | 3.7491 | 1 | 0.1937 | `[24.64, 156.35, 132.91, 265.46]` | (78.8, 210.9) |

人工标注对照：第 200 帧无球（空标注）；第 225 帧标注框 `[26, 156, 133, 265]`，中心 (79.5, 210.5)，与模型中心基本重合。

- `replay-200-225.jsonl`：逐帧记录（帧号、时间、检测列表、中心点）
- `replay-200-225.csv`：表格版
- `replay-200-225.png`：两帧并列截图

## 前置条件

`dataset-curated/` 与 `yolo11n.pt` 被 `.gitignore` 忽略，复跑前在仓库根目录执行一次：

```bash
python lab.py prepare
```
