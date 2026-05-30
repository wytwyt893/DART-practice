# Stage 4: Real-format Mock Router

## 实验目的

本实验用于验证当前项目能否从纯 synthetic 数据，过渡到官方 TableDART 风格的真实 `jsonl` 数据格式。

这一阶段暂时不调用真实大模型，也不复现完整 TableDART benchmark。当前目标是先打通工程闭环：

```text
官方真实格式数据
-> Dataset
-> DataLoader
-> MiniDynamicSelector
-> MLPRouter
-> train / checkpoint / eval
```

## 配置

配置文件：

```text
configs/router_real_format_mock.yaml
```

关键设置：

```text
data_path: TableDART-Official/TableDART/data/test/test_data.jsonl
samples_per_category: 64
text_dim: 4096
image_dim: 4096
question_dim: 1920
hidden_dim: 256
num_routes: 3
epochs: 10
train_batch_size: 64
val_batch_size: 128
lr: 0.001
```

路由器结构：

```text
text_feature     [B, 4096]
image_feature    [B, 4096]
question_feature [B, 1920]
        -> concat
combined_feature [B, 10112]
        -> Linear(10112, 256)
        -> ReLU
        -> Dropout(0.1)
        -> Linear(256, 3)
        -> route_logits [B, 3]
```

## 数据检查

运行命令：

```powershell
python scripts/inspect_real_format_dataset.py
```

当前读取结果：

```text
成功读取样本数: 448

WTQ_for_TQA: 64
TabFact_for_TFV: 64
FeTaQA_for_TQA: 64
TAT-QA_for_TQA: 64
HiTab_for_TQA: 64
TABMWP_for_TQA: 64
InfoTabs_for_TFV: 64
```

伪路由标签分布：

```text
route 0 / 偏文本路径: 128
route 1 / 偏图像或表格结构路径: 128
route 2 / 偏融合或问题路径: 192
```

说明：当前 route label 是根据 `category` 构造的 mock label，用于验证训练流程，不是论文中的真实 router supervision。

## 训练结果

运行命令：

```powershell
python train_real_format_mock.py
```

关键训练日志：

```text
Epoch 01 | train_loss=0.7934 train_acc=0.7235 | val_loss=0.4051 val_acc=0.9333
Epoch 02 | train_loss=0.0046 train_acc=1.0000 | val_loss=0.2173 val_acc=0.9444
Epoch 03 | train_loss=0.0001 train_acc=1.0000 | val_loss=0.1646 val_acc=0.9667
Epoch 04 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.1441 val_acc=0.9778
Epoch 10 | train_loss=0.0000 train_acc=1.0000 | val_loss=0.1245 val_acc=0.9667
```

checkpoint 输出：

```text
checkpoints/router_real_format_mock_last.pth
checkpoints/router_real_format_mock_best.pth
```

## 独立评估结果

运行命令：

```powershell
python eval_real_format_mock.py
```

输出：

```text
Loaded checkpoint: checkpoints\router_real_format_mock_best.pth
Checkpoint epoch: 4
Checkpoint val_acc: 0.9777777777777777
独立评估结果 | val_loss=0.1441 val_acc=0.9778
```

这说明 best checkpoint 可以被独立加载，并且能够复现训练阶段记录的验证集结果。

## 当前结论

Stage 4 已完成：

- 官方真实格式 `jsonl` 可以被项目 Dataset 正确读取。
- 真实字段 `question / table / answer / image / category` 可以保留下来。
- 变长表格字段可以通过 `real_format_collate_fn` 正常组成 batch。
- 三路 mock feature 可以对接 `MiniDynamicSelector`。
- 训练、保存 checkpoint、独立评估均已跑通。

## 局限

- 当前 feature 仍然是随机 mock feature，不是真实大模型或 encoder 提取的 embedding。
- 当前 route label 是 category-based mock label，不是 TableDART 根据 expert 正确性构造的 soft target。
- 当前 loss 是普通 `CrossEntropyLoss`，还没有对齐论文中的 task loss / resource loss。
- 当前结果只说明工程流程有效，不能解释为真实 TableDART benchmark 性能。

## 下一步

- 阅读并对齐官方 `dynamic_selector.py` / `gating_network.py` 中的训练目标。
- 设计真实 feature extraction 流程，把 mock feature 替换为真实 encoder embedding。
- 研究 expert/decoder 输出如何生成路由监督信号。
- 在真实 feature 和真实 supervision 接入后，再进行更接近论文设置的消融实验。
