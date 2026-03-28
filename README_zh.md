# TopoGO — 图像结检测与拓扑分析

[English Version](./README.md)

## 项目简介

本项目旨在识别结（Knot）图像并计算其拓扑不变量——亚历山大多项式（Alexander Polynomial）。项目中实现了两种核心检测算法流程：

### 流程 A — 滑动窗口法 (Sliding Window Method)
通过滑动窗口扫描标记后的图像来识别交叉点。

- **单图演示：** `demo_window.py`
- **批量处理：** `batch_window.py`

### 流程 B — 中心线法 (Centerline Method)
提取图像骨架的中心线，通过识别端点并结合几何特征来检测交叉点。

- **单图演示：** `demo_centerline.py`
- **批量处理：** `batch_centerline.py`

## 处理流程

```
 图像加载 → 二值化 → 分割 → 标记 → 分类 → 交叉点检测 → 亚历山大多项式
  (s0)     (s1)     (s2)   (s3)   (s4)   (s5a / s5b)      (s6)
```

`src/` 目录下的源码模块按处理步骤编号：

| 模块 | 步骤 | 描述 |
|---|---|---|
| `0_io.py` | 0 | 图像加载与保存 |
| `1_binarize.py` | 1 | 基于阈值的二值化 |
| `2_segment.py` | 2 | 连通分量标记 |
| `3_label_utils.py` | 3 | 标签排列、分离与合并 |
| `4_classify.py` | 4 | 线段分类 |
| `5a_cross_detect.py` | 5a | 滑动窗口交叉点检测（流程 A） |
| `5b_centerline.py` | 5b | 基于中心线的交叉点检测（流程 B） |
| `5b_thinning.py` | 5b | Zhang-Suen 细化算法（参考实现） |
| `6_alexander.py` | 6 | 亚历山大多项式计算 |
| `viz.py` | — | 可视化工具库 |

## 环境配置

```bash
pip install -r requirements.txt
```

## 参考文献与资源

- `TopoGO_Knot-Detection-of-Images.pdf` — 项目论文
- `SUSTech_DIP_Poster_*.pdf` — 会议海报
- 输入图像来源于 [Knot Atlas](https://katlas.org/)

---

![Poster](analysis/SUSTech_DIP_Poster_Eason.png)
![Paper](analysis/Paper_preview.png)

---

> **TODO:** 证明与其他拓扑不变量的转换关系；探索计算亚历山大多项式的替代方法。
