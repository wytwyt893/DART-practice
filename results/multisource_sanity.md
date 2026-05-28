# Multi-source MiniDynamicSelector Sanity Check

## 实验目的

本实验是 TableDART 动态路由复现实验的 Stage 2。

Stage 1 中，模型输入是一整块 synthetic feature：

```text
[B, 10112] -> MLPRouter -> route_logits
```

Stage 2 将输入拆成三路 synthetic gate feature：

```text
text_feature     [B, 4096]
image_feature    [B, 4096]
question_feature [B, 1920]
        ↓ concat
combined_feature [B, 10112]
        ↓
MLPRouter
        ↓
route_logits     [B, 3]
```

该结构用于模拟官方 TableDART 中的：

```python
text_gate_feats + vlm_gate_feats + q_embeddings_for_gate
```

当前实验重点不是追求真实 benchmark 指标，而是验证 multi-source dynamic routing 的数据流、模型流、训练流和 checkpoint/eval 闭环。

## 模型结构

当前模型为 `MiniDynamicSelector`：

```text
MiniDynamicSelector
├── concat(text_feature, image_feature, question_feature)
└── MLPRouter
    ├── Linear(10112, 256)
    ├── ReLU
    ├── Dropout(0.1)
    └── Linear(256, 3)
```

与官方代码的对应关系：

| 当前项目 | 官方 TableDART |
|---|---|
| `models/mini_dynamic_selector.py` | `models/dynamic_selector.py` 的简化版 |
| `models/router.py` | `models/gating_network.py` |
| `text_feature` | `text_gate_feats` |
| `image_feature` | `vlm_gate_feats` |
| `question_feature` | `q_embeddings_for_gate` |
| `route_label` | 当前 synthetic 标签，暂时代替 expert-score soft target |

## 数据构造

配置文件：

```text
configs/router_multisource_sanity.yaml
```

主要参数：

| 项目 | 设置 |
|---|---:|
| 样本数 | 1200 |
| text_dim | 4096 |
| image_dim | 4096 |
| question_dim | 1920 |
| 拼接后总维度 | 10112 |
| 路由数 | 3 |
| train / val | 0.8 / 0.2 |
| hidden_dim | 256 |
| dropout | 0.1 |
| learning rate | 0.001 |
| epochs | 10 |

信号注入规则：

```text
route 0 -> text_feature 的部分维度增强
route 1 -> image_feature 的部分维度增强
route 2 -> question_feature 的部分维度增强
```

因此模型需要学习：

```text
哪一路来源的 feature 信号更强，就选择对应 route。
```

## 运行命令

训练：

```powershell
python train_multisource.py
```

独立评估：

```powershell
python eval_multisource.py
```

## 实验结果

训练日志：

```text
Epoch 01 | train_loss=0.4233 train_acc=0.8792 | val_loss=0.0754 val_acc=0.9958
Epoch 02 | train_loss=0.0009 train_acc=1.0000 | val_loss=0.0259 val_acc=0.9958
Epoch 03 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0214 val_acc=0.9958
Epoch 04 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0205 val_acc=0.9958
Epoch 05 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0203 val_acc=0.9958
Epoch 06 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0202 val_acc=0.9958
Epoch 07 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0201 val_acc=0.9958
Epoch 08 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0201 val_acc=0.9958
Epoch 09 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0200 val_acc=0.9958
Epoch 10 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0199 val_acc=0.9958
```

独立评估结果：

```text
Loaded checkpoint from epoch 1
Checkpoint val_acc: 0.9958333333333333
Eval result | val_loss=0.0754 val_acc=0.9958
```

## 结果解读

当前 multi-source 版本在 synthetic setting 下可以快速收敛，说明：

- `MultiSourceSyntheticDataset` 能正常返回 dict batch。
- `MiniDynamicSelector` 能正确拼接三路 feature。
- `MLPRouter` 能根据三路来源中的增强信号完成路由分类。
- `train_multisource.py` 和 `eval_multisource.py` 已完成完整实验闭环。

但需要注意：

- 该结果不代表真实 TableDART benchmark 性能。
- 当前高准确率主要来自人工构造的强 synthetic signal。
- 该实验的主要价值是验证结构和流程，而不是证明方法优于 baseline。

## 与 Stage 1 的区别

| 阶段 | 输入 | 模型 | 目的 |
|---|---|---|---|
| Stage 1 | 单一 `[B, 10112]` feature | `MLPRouter` | 跑通基础训练闭环 |
| Stage 2 | `text/image/question` 三源 feature | `MiniDynamicSelector + MLPRouter` | 对齐 TableDART dynamic routing 数据流 |

## 下一步

- 增加 multi-source 结构可视化图。
- 做 feature ablation，例如去掉 `question_feature`，比较 `text+image+question` 与 `text+image`。
- 继续阅读官方 `dynamic_selector.py`，重点关注 task loss 和 resource loss。
- 后续将 synthetic route labels 替换为更接近官方 expert score 的 soft target。
