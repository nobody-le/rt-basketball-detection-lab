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
| 耗时 | 84.22 ms |

- `rt-100.png`：带框结果图
- `result.json`：完整运行记录（含权重校验值、依赖版本、复跑命令）

## 前置条件

`dataset-curated/` 与 `yolo11n.pt` 被 `.gitignore` 忽略，复跑前在仓库根目录执行一次：

```bash
python lab.py prepare
```
