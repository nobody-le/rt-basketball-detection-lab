# 任务五 · 来源交叉修正与 Jackson 独立测试

## 两处来源交叉

| # | 帧 | 错误位置 | 应属 | 影响 |
|---:|---|---|---|---|
| 1 | `rt-125` | train | val | RT 来源同时出现在训练和验证；验证集不再是纯 RT，训练也提前看到了 RT 画面 |
| 2 | `jackson-011` | val | test | Jackson 来源同时出现在验证和测试；`best.pt` 选型会看到测试来源，测试集不再完全留出 |

错误划分下总数为 train 34、val 14、test 7；修正后为 train 33、val 14、test 8。

## 修正后的分组（按视频来源）

| 来源 | 帧数 | 修正后 split | 错误划分 |
|---|---:|---|---|
| PLOS 研究录像（`plos01/02/03/06/07/09`） | 33 | train | 全部 train |
| RT 练习视频（`rt.mp4`） | 14 | val | `rt-125` 在 train，其余 13 在 val |
| Jackson 罚球视频 | 8 | test | `jackson-011` 在 val，其余 7 在 test |

明细见 `grouping.csv`。修正原则是按来源整组划分：同一视频来源只落在一个 split，避免同源画面跨 train/val/test。

## Jackson 独立测试

设置：预训练 `yolo11n.pt`、CPU、`imgsz=416`、`conf=0.10`、仅 `sports ball`、匹配 IoU ≥ 0.50。测试对象是修正后留出的 8 张 Jackson 罚球图。

| 帧 | 文件 | 分组 | 标注框 `xyxy` | 检测数 | 最高分 | 最佳 IoU | 匹配 | 需复核 | 原因 |
|---:|---|---|---|---:|---:|---:|---|---|---|
| 3 | `jackson-003.jpg` | small | `[584, 785, 694, 887]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 11 | `jackson-011.jpg` | small | `[626, 1139, 720, 1233]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 19 | `jackson-019.jpg` | small | `[605, 875, 707, 969]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 27 | `jackson-027.jpg` | small | `[631, 879, 733, 981]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 35 | `jackson-035.jpg` | small | `[694, 811, 805, 896]` | 1 | 0.2768 | 0.0000 | 否 | **是** | unmatched_extra_box |
| 43 | `jackson-043.jpg` | small | `[648, 495, 767, 597]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 51 | `jackson-051.jpg` | edge | `[767, 0, 852, 73]` | 0 | — | 0.0000 | 否 | **是** | miss |
| 59 | `jackson-059.jpg` | empty | 无球 | 0 | — | — | 不适用 | 否 | ok |

汇总：有球匹配 **0 / 7**，无球误报 **0 / 1**，额外框 **1**。

### 需要人工复核的画面

**7 张有球帧全部需要复核**：`003`、`011`、`019`、`027`、`035`、`043`、`051`。模型在 416/0.10 下没有框中任何一颗球。

其中 `035` 还有另一个问题：模型在左下角输出 `[35.92, 1321.71, 97.30, 1394.76]`、置信度 0.2768 的框，与标注球位无重叠，是额外的误检，复核时要同时确认该位置是什么。

`059` 的标注为无球，模型也没有输出，二者一致，不需要复核。

## 人工框与预测框对照

每帧一张 `jackson-<帧号>-compare.jpg`：绿色为人工框，红色为预测框，页眉标出帧号、IoU 和匹配结论。`jackson-8-grid.jpg` 是 8 帧总览。

| 帧 | 对照图 | IoU | 预测框 |
|---:|---|---:|---|
| 3 | `jackson-003-compare.jpg` | 0.0000 | 无 |
| 11 | `jackson-011-compare.jpg` | 0.0000 | 无 |
| 19 | `jackson-019-compare.jpg` | 0.0000 | 无 |
| 27 | `jackson-027-compare.jpg` | 0.0000 | 无 |
| 35 | `jackson-035-compare.jpg` | 0.0000 | 1 个，conf 0.2768，与标注无重叠 |
| 43 | `jackson-043-compare.jpg` | 0.0000 | 无 |
| 51 | `jackson-051-compare.jpg` | 0.0000 | 无 |
| 59 | `jackson-059-compare.jpg` | —（无球） | 无 |

`comparison-index.csv` 记录每帧的人工框、预测框坐标、置信度、IoU、匹配与复核标志。

## 复跑

```bash
env -u PYTHONPATH .venv/bin/python submission/task05/run_test.py
env -u PYTHONPATH .venv/bin/python submission/task05/draw_comparison.py
```

输出：`jackson-test.jsonl`（逐帧完整检测）、`test-frames.csv`（上表）、`test-summary.json`（汇总与复核帧清单）、`grouping.csv`（来源分组）、`jackson-*-compare.jpg` / `jackson-8-grid.jpg` / `comparison-index.csv`（对照图与索引）。
