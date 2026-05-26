# DART Practice

**5.25 更新**

## Project Goal

- 基于UMMs统一多模态模型路线 + 动态路由网络TableDART进行的创新实验尝试

- 整体思路为 **“多路解耦道路 + UMMs统一多模态架构 +输入端做类似Chameleon的早期离散融合（Early-fusion）机制，将表格的复杂像素与排版彻底转化为统一的视觉 Token + 三路decoder包装成三个独立的tools交给多模态Agent框架（如LangGraph）”**

- 后续会根据UMMs路线进一步调研并优化encoder方面的创新尝试

- 5.25：补充真实参数量、日志、检查点、配置文件读取和独立评估脚本

## Current Stage

- 当前完成 synthetic sanity check，不是完整 TableDART
- 已完成 **模拟数据集** - **基础双层 MLP Router** - **配置驱动训练脚本** - **日志与检查点保存** - **独立评估脚本** 的最小实验闭环
- 当前 `train.py` 已支持从 `configs/router_sanity.yaml` 读取主要实验参数
- 当前 `eval.py` 已支持加载 `router_sanity_best.pth` 并复现验证集指标

## Environment

本项目实验在以下硬件与软件环境下完成。为保证实验结果的可复现性，建议尽量使用相同或相近的依赖版本。
### 1. 硬件环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 可用显卡数量 | 1 |
| CUDA 是否可用 | True |

### 2. 软件依赖

| 依赖库 | 版本 |
|---|---|
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| NumPy | 2.2.5 |
| Weights & Biases | 0.26.1 |
| tqdm | 4.67.3 |

### 3. CUDA 配置

| 项目 | 配置 |
|---|---|
| CUDA 支持 | 已启用 |
| CUDA 版本 | 12.1 |
| 当前显卡 | NVIDIA GeForce RTX 4060 Laptop GPU |

### 4. 环境检查脚本

可以使用以下脚本检查当前运行环境是否与实验环境一致：

```python
import torch
import torchvision
import numpy as np
import wandb
import tqdm

print("PyTorch 版本:", torch.__version__)
print("Torchvision 版本:", torchvision.__version__)
print("NumPy 版本:", np.__version__)
print("WandB 版本:", wandb.__version__)
print("tqdm 版本:", tqdm.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("可用显卡数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前显卡型号:", torch.cuda.get_device_name(0))
```

### 5. 复现说明
本项目实验基于单张 NVIDIA GeForce RTX 4060 Laptop GPU 完成，并启用了 CUDA 加速。  
为尽可能复现实验结果，建议保持 PyTorch、Torchvision、NumPy 等核心依赖版本与上表一致。

## How to Run

### 1. 训练 Router Sanity Check

运行当前的 TableDART-router-like synthetic sanity check 训练实验：

```powershell
python train.py
```

该命令会读取以下配置文件：

```text
configs/router_sanity.yaml
```

随后在合成的 10112 维特征上训练一个双层 MLP 路由器，并将每个 epoch 的训练和验证指标写入：

```text
logs/router_sanity_log.txt
```

训练过程中会保存两个 checkpoint：

```text
checkpoints/router_sanity_last.pth
checkpoints/router_sanity_best.pth
```

### 2. 独立评估 Best Checkpoint

训练完成后，可以使用 `eval.py` 单独加载最佳 checkpoint，并在验证集上重新计算指标：

```powershell
python eval.py
```

该命令会读取：

```text
checkpoints/router_sanity_best.pth
```

并输出 checkpoint 中保存的 epoch、验证集准确率，以及重新评估得到的 `val_loss` 和 `val_acc`。

### 3. 绘制训练曲线

如果需要根据训练日志绘制 loss 和 accuracy 曲线，可以运行：

```powershell
python plot_training_curves.py --log-file logs/router_sanity_log.txt --out logs/router_sanity_curves.png
```

生成的曲线图会保存在：

```text
logs/router_sanity_curves.png
```

## Current Result

当前实验配置记录在：

```text
configs/router_sanity.yaml
```

主要实验设置如下：

| 项目 | 配置 |
|---|---|
| 输入特征维度 | 10112 |
| 隐藏层维度 | 256 |
| 路由类别数 | 3 |
| Dropout | 0.1 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 训练轮数 | 10 |
| 训练 batch size | 64 |
| 验证 batch size | 128 |
| 最佳 checkpoint | checkpoints/router_sanity_best.pth |

当前一次训练运行中的最佳验证集准确率为：

```text
val_acc = 0.9458
```

最佳 checkpoint 出现在第 2 个 epoch：

```text
Best checkpoint epoch = 2
```

使用 `eval.py` 加载 `router_sanity_best.pth` 后，重新评估得到：

```text
val_loss = 0.1363
val_acc  = 0.9458
```

这说明当前实验已经完成了从训练、保存 checkpoint 到独立加载并复现验证集结果的闭环。

需要注意：该结果只说明当前 synthetic 数据上的训练流程是可运行、可学习的；它不是完整 TableDART 在真实 benchmark 上的复现结果。

## Next Steps

后续计划包括：

- 将 `train.py` 和 `eval.py` 中重复出现的 `load_config`、`set_seed`、`evaluate` 等函数逐步抽取到 `utils/` 目录中。
- 将代码中的普通分类命名进一步改成路由语义命名，例如 `num_routes`、`route_labels`、`route_logits`。
- 增加命令行参数，使 `train.py` 和 `eval.py` 可以指定不同的 config 文件。
- 将当前 synthetic 特征逐步替换为更接近真实表格/多模态任务的特征。
- 阅读和整理原始 TableDART baseline，优先确定最适合本科生复现的 router 模块部分。
- 在完成 router 模块复现后，再考虑更复杂的 early-fusion、多路 decoder 和 Agent/tool 包装。
