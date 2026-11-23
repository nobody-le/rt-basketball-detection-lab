# 任务一 · 环境核验

核验项目 `.venv` 的 Python 路径、课程锁定依赖版本，以及 PyTorch 是否为 CPU 版。

## 复跑入口

在仓库根目录执行：

```bash
env -u PYTHONPATH .venv/bin/python submission/task01/check_env.py | tee submission/task01/environment.log
```

`env -u PYTHONPATH` 用于隔离本机 ROS 2 的 `site-packages`，避免其插到 venv 之前。

## 本次结果

- `environment.log`：`PASS: 13/13 checks`，退出码 0
- Python：`/home/nobody-le/projects/rt-basketball-detection-lab/.venv/bin/python`（3.12.3）
- torch `2.6.0+cpu`、torchvision `0.21.0+cpu`、ultralytics `8.3.232`、opencv-python `4.11.0.86`、numpy `1.26.4`
- CPU 核验：`torch.version.cuda=None`、`torch.cuda.is_available()=False`、`device_count=0`、矩阵运算结果在 `cpu`

> PyTorch CPU 索引安装后的包元数据版本带 `+cpu` 本地后缀，脚本按 `2.6.0` / `2.6.0+cpu` 两种写法视为同一锁定版本。
