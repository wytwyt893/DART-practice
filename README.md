# DART Practice

本项目是围绕 TableDART 动态路由思想展开的本科科研复现实验练习。当前目标不是直接复现完整 TableDART benchmark，而是先从轻量级、可控的 synthetic setting 出发，逐步搭建可复现、可替换、可扩展的动态路由实验底座。

## Project Goal

长期方向：

- 以 TableDART 的动态路由机制为 baseline，学习并复现表格多模态理解中的 Text / Image / Fusion 路由选择流程。
- 后续尝试围绕多源 gate feature、cost-aware routing、uncertainty-aware routing 或 early-fusion gate 进行小规模优化。
- 逐步从 synthetic feature 过渡到 official-style feature，再考虑真实数据和真实专家模型。

当前阶段定位：

- 当前实验仍是 synthetic sanity check，不是完整 TableDART 真实 benchmark 复现。
- 当前重点是训练基本功和科研工程闭环：`Dataset / DataLoader / train loop / eval / checkpoint / config / log / visualization`。

## Current Stage

### Stage 1: Single-source MLPRouter

第一阶段完成了一个单输入特征版本：

```text
SyntheticDataset
-> MLPRouter
-> train.py
-> eval.py
```

该阶段模拟：

```text
[batch_size, 10112] feature -> Linear -> ReLU -> Dropout -> Linear -> 3 route logits
```

已完成能力：

- 基于 synthetic route labels 构造可学习信号。
- 完成 config-driven 训练流程。
- 完成 best / last checkpoint 保存。
- 完成独立 `eval.py` 加载 best checkpoint 并复现验证集结果。
- 完成 dropout ablation 初步实验。

### Stage 2: Multi-source MiniDynamicSelector

第二阶段将单一 feature 升级为更接近官方 TableDART 的多源 gate feature：

```text
MultiSourceSyntheticDataset
-> MiniDynamicSelector
-> train_multisource.py
-> eval_multisource.py
```

当前 multi-source 数据流：

```text
text_feature     [B, 4096]
image_feature    [B, 4096]
question_feature [B, 1920]
        ↓ concat(dim=1)
combined_feature [B, 10112]
        ↓
MLPRouter
        ↓
route_logits     [B, 3]
```

这一步对齐官方 TableDART 中的核心结构：

```python
combined_gate_features = torch.cat(
    [text_gate_feats, vlm_gate_feats, q_embeddings_for_gate],
    dim=1,
)
gate_logits = self.gating_network(combined_gate_features)
```

## Environment

| Item | Version / Config |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| CUDA available | True |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| NumPy | 2.2.6 |
| WandB | 0.26.1 |
| tqdm | 4.67.3 |

环境检查脚本：

```powershell
python test_env.py
```

## How to Run

### 1. Single-source Router Sanity Check

训练：

```powershell
python train.py
```

或显式指定配置：

```powershell
python train.py --config configs/router_sanity.yaml
```

评估 best checkpoint：

```powershell
python eval.py
```

或：

```powershell
python eval.py --config configs/router_sanity.yaml
```

主要输出：

```text
logs/router_sanity_log.txt
checkpoints/router_sanity_best.pth
checkpoints/router_sanity_last.pth
```

### 2. Multi-source MiniDynamicSelector

训练：

```powershell
python train_multisource.py
```

评估 best checkpoint：

```powershell
python eval_multisource.py
```

主要输出：

```text
logs/router_multisource_sanity_log.txt
checkpoints/router_multisource_sanity_best.pth
checkpoints/router_multisource_sanity_last.pth
```

### 3. Dropout Ablation Summary

生成 dropout 消融实验汇总表：

```powershell
python scripts/summarize_dropout_ablation.py
```

实验记录：

```text
results/dropout_ablation.md
```

### 4. Meeting Figures

生成组会汇报用图片：

```powershell
python scripts/generate_meeting_figures.py
python scripts/plot_mlp_router_architecture.py
```

图片输出目录：

```text
results/figures/
```

## Current Results

### Stage 1 Result

配置文件：

```text
configs/router_sanity.yaml
```

最佳验证结果：

```text
Best checkpoint epoch = 2
val_loss = 0.1363
val_acc  = 0.9458
```

说明：

- 该结果证明 single-source synthetic router 的训练、保存和独立评估流程已跑通。
- 该结果不代表真实 TableDART benchmark 性能。

### Stage 2 Result

配置文件：

```text
configs/router_multisource_sanity.yaml
```

训练输出：

```text
Epoch 01 | train_loss=0.4233 train_acc=0.8792 | val_loss=0.0754 val_acc=0.9958
Epoch 10 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.0199 val_acc=0.9958
```

独立评估输出：

```text
Loaded checkpoint from epoch 1
Checkpoint val_acc: 0.9958333333333333
Eval result | val_loss=0.0754 val_acc=0.9958
```

说明：

- multi-source 版本已经完成 `train -> checkpoint -> eval -> reproduce result` 闭环。
- 当前高准确率来自人工构造的 synthetic signal，不能解读为真实任务性能提升。
- 该阶段的主要意义是：数据流结构从单一 feature 升级为 `text/image/question` 三源 gate feature，开始对齐官方 `dynamic_selector.py`。

## Project Structure

```text
DART-UMMs
├── configs/       实验配置文件
├── data/          SyntheticDataset / MultiSourceSyntheticDataset
├── models/        MLPRouter / MiniDynamicSelector
├── utils/         config / seed / metrics 等工具
├── scripts/       汇总与可视化脚本
├── results/       实验记录与图表
├── train.py       single-source 训练入口
├── eval.py        single-source 评估入口
├── train_multisource.py
└── eval_multisource.py
```

## Notes

- `TableDART-Official/` 是本地参考仓库，已加入 `.gitignore`，不提交到本项目。
- `checkpoints/` 和 `logs/` 默认也不提交，用于本地实验记录。
- 当前代码优先服务于学习和复现实验闭环，后续稳定后会逐步把通用函数抽到 `utils/`。

## Next Steps

- 新增 `results/multisource_sanity.md`，记录 Stage 2 实验目的、结构、结果和局限。
- 为 multi-source 结构生成更直观的结构图。
- 做 Stage 3 feature ablation：例如 `text + image + question` vs `text + image only`。
- 逐步把 synthetic feature 替换为更接近官方 TableDART 的 gate feature。
- 阅读并对齐官方 `models/dynamic_selector.py` 中 task loss / resource loss 的训练逻辑。

## Stage 4 Update: Real-format Mock Router

当前已经完成 Stage 4 的真实格式数据接入闭环。该阶段开始读取官方 TableDART 仓库中的真实 `jsonl` 数据格式，但暂时仍使用 mock feature，不调用真实大模型。

当前数据流：

```text
TableDART official-style jsonl
-> RealFormatMockFeatureDataset
-> real_format_collate_fn
-> MiniDynamicSelector
-> MLPRouter
-> train_real_format_mock.py
-> eval_real_format_mock.py
```

已完成内容：

- 读取官方真实格式字段：`question_id / category / question / table / answer / image`。
- 按任务类别均衡采样：当前每个 category 取 64 条样本，共 448 条。
- 将真实格式样本包装成三路 mock feature：`text_feature / image_feature / question_feature`。
- 使用 `real_format_collate_fn` 处理真实表格的变长字段，保证 DataLoader 可以正常组 batch。
- 完成 `train -> checkpoint -> eval` 独立闭环。

运行命令：

```powershell
python scripts/inspect_real_format_dataset.py
python train_real_format_mock.py
python eval_real_format_mock.py
```

当前结果：

```text
Loaded checkpoint: checkpoints\router_real_format_mock_best.pth
Checkpoint epoch: 4
Checkpoint val_acc: 0.9777777777777777
独立评估结果 | val_loss=0.1441 val_acc=0.9778
```

阶段边界：

- 当前高准确率来自 mock feature 中注入的可学习信号，不代表真实 TableDART benchmark 性能。
- 当前 route label 是根据 `category` 构造的伪标签，不是论文中根据 expert 正确性构造的 soft target。
- 下一步需要把 mock feature 替换为真实 encoder embedding，并继续对齐 TableDART 的 task loss / resource loss。
