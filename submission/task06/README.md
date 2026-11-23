# 任务六 · 固定 416 ONNX 与 PyTorch / ONNX Runtime 对照

回看接口只接收 416×416 输入，因此把预训练 `yolo11n.pt` 导出为固定尺寸 ONNX，并用仓库的 `compare_onnx.py` 在同一开发机上核对两个后端。

## 导出配置

| 项 | 值 |
|---|---|
| 源权重 | `yolo11n.pt` |
| 导出文件 | `yolo11n.onnx` |
| SHA-256 | `5c3094a18d854db2e5f33e74e836fad2e869ce235b81524ffb12aa24015bcf3c` |
| 大小 | 10,604,541 bytes |
| imgsz / batch | 416 / 1 |
| dynamic / simplify | false / false |
| device | CPU |
| opset | 19 |
| 输入 | `images`，`[1, 3, 416, 416]`，float32 |
| 输出 | `output0`，`[1, 84, 3549]` |
| 版本 | Ultralytics 8.3.232、torch 2.6.0+cpu、onnx 1.19.1、onnxruntime 1.32.0 |

导出命令：

```bash
env -u PYTHONPATH .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=416, batch=1, dynamic=False, simplify=False, device='cpu')"
```

导出时的 Ultralytics 日志确认：`input shape (1, 3, 416, 416)`，保存为 `yolo11n.onnx`。完整配置见 `export_config.json`。

## 输入尺寸记录

`onnx-shape.json` 由 ONNX 图和 ONNX Runtime 会话读取：

- 图输入 `images`：`[1, 3, 416, 416]`
- ONNX Runtime 输入：`[1, 3, 416, 416]`，`tensor(float)`
- 图输出 `output0`：`[1, 84, 3549]`
- provider：`CPUExecutionProvider`

`compare_onnx.py` 的前向钩子记录 PyTorch 实际输入为 `[1, 3, 416, 416]`（三张图各一次），ONNX 固定输入为 `[1, 3, 416, 416]`。两边都使用 `rect=false`，即相同的等比缩放与补边；阈值 0.10，仅 `sports ball`，输出统一换算回原图像素坐标。

## 三张图的逐图对照

| 帧 | PyTorch | ONNX Runtime | 一致 |
|---:|---|---|---|
| 100 | `sports ball`，conf **0.638958**，框 `[140.1268, 391.1578, 260.0394, 516.3745]` | `sports ball`，conf **0.638958**，框 `[140.1268, 391.1577, 260.0394, 516.3745]` | 是 |
| 150 | 空结果 | 空结果 | 是 |
| 200 | 空结果 | 空结果 | 是 |

数值差：第 100 帧最大框坐标差 `6.1 × 10⁻⁵ px`，最大置信度差 `5.96 × 10⁻⁸`；150、200 帧两边都没有检测。

明细见 `backend-table.csv` 与 `backend-comparison.json`。

## 切换判断

| 条件 | 结果 |
|---|---|
| 固定输入尺寸 416×416 | 满足 |
| 框一致（容差 0.01 px） | 满足，最大差 0.000061 px |
| 类别一致 | 满足，均为 `sports ball` |
| 置信度一致（容差 1e-4） | 满足，最大差 5.96e-8 |
| 空结果一致 | 满足，150/200 两边均为空 |

结论：**在这三张核对图上达到切换 ONNX Runtime 的条件**。两边的预处理、输入尺寸、框、类别、置信度和空结果都对齐，差异仅为浮点舍入量级。

适用边界：核对样本只有 3 张，其中 2 张是空结果；它证明了导出尺寸和后端一致性，不等同于覆盖所有场景的精度验收。切换前建议再在完整验证/测试集上跑一次同样的双后端对照。

## 文件与复跑

```text
submission/task06/
├── export_config.json          # 导出参数与校验值
├── onnx-shape.json             # ONNX/ORT 输入输出尺寸
├── backend-comparison.json     # compare_onnx.py 原始输出
├── backend-comparison.log      # 运行日志
├── backend-table.csv           # 三张图逐项对照
├── switch-decision.json        # 条件检查与结论
├── summarize.py
└── README.md
```

```bash
# 1) 导出（生成 yolo11n.onnx，仓库忽略 *.onnx）
env -u PYTHONPATH .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=416, batch=1, dynamic=False, simplify=False, device='cpu')"

# 2) 双后端核对
env -u PYTHONPATH .venv/bin/python compare_onnx.py --onnx ./yolo11n.onnx --output submission/task06/backend-comparison.json

# 3) 汇总与切换条件
env -u PYTHONPATH .venv/bin/python submission/task06/summarize.py
```
