# Dropout Ablation Experiment

## 实验目的

本实验用于观察 `MLPRouter` 中不同 dropout 设置对 synthetic route-label sanity check 的影响。

当前实验仍然使用合成路由标签，不是完整 TableDART benchmark 复现。该实验的主要目的不是证明某个 dropout 设置在真实任务上最优，而是练习科研实验中的控制变量、日志记录、checkpoint 保存和结果汇总流程。

## 实验设置

本组实验只改变 `model.dropout`，其他参数保持一致。

| 项目 | 设置 |
|---|---|
| 输入特征维度 | 10112 |
| 隐藏层维度 | 256 |
| 路由数 | 3 |
| 训练样本数 | 1200 |
| 训练/验证划分 | 0.8 / 0.2 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 训练轮数 | 10 |
| 随机种子 | 42 |

对比配置如下：

| Dropout | Seed | Config | Log | Best Checkpoint |
|---:|---:|---|---|---|
| 0.0 | 42 | `configs/router_dropout_0.yaml` | `logs/router_dropout_0_log.txt` | `checkpoints/router_dropout_0_best.pth` |
| 0.1 | 42 | `configs/router_sanity.yaml` | `logs/router_sanity_log.txt` | `checkpoints/router_sanity_best.pth` |
| 0.3 | 42 | `configs/router_dropout_03.yaml` | `logs/router_dropout_03_log.txt` | `checkpoints/router_dropout_03_best.pth` |
| 0.0 | 123 | `configs/router_dropout_0_seed123.yaml` | `logs/router_dropout_0_seed123_log.txt` | `checkpoints/router_dropout_0_seed123_best.pth` |
| 0.3 | 123 | `configs/router_dropout_03_seed123.yaml` | `logs/router_dropout_03_seed123_log.txt` | `checkpoints/router_dropout_03_seed123_best.pth` |

## 实验结果

| Dropout | Seed | Best Epoch | Best Val Loss | Best Val Acc |
|---:|---:|---:|---:|---:|
| 0.0 | 42 | 2 | 0.1330 | 0.9500 |
| 0.1 | 42 | 2 | 0.1363 | 0.9458 |
| 0.3 | 42 | 2 | 0.1397 | 0.9542 |
| 0.0 | 123 | 2 | 0.1648 | 0.9458 |
| 0.3 | 123 | 2 | 0.1586 | 0.9417 |

## 结果观察

在当前 synthetic router sanity-check 设置下，`dropout=0.3` 在 `seed=42` 时取得了最高的 best validation accuracy，达到 `0.9542`。

不过加入 `seed=123` 后，`dropout=0.0` 的 best validation accuracy 为 `0.9458`，高于 `dropout=0.3` 的 `0.9417`。这说明当前 dropout 排序会随随机种子变化，不能得出 `dropout=0.3` 稳定更优的结论。

因此，当前更稳妥的观察是：在该 synthetic route-label 设置下，dropout 对验证准确率的影响较小，且不同 seed 下排序不稳定。

由于当前数据是人工构造的 synthetic route labels，不能据此得出 dropout 在真实 TableDART 或真实表格多模态任务中更优的结论。后续需要在更多随机种子、更真实的特征或真实 benchmark 上进一步验证。

## 复现实验命令

```powershell
python train.py --config configs/router_dropout_0.yaml
python eval.py --config configs/router_dropout_0.yaml

python train.py --config configs/router_sanity.yaml
python eval.py --config configs/router_sanity.yaml

python train.py --config configs/router_dropout_03.yaml
python eval.py --config configs/router_dropout_03.yaml

python train.py --config configs/router_dropout_0_seed123.yaml
python eval.py --config configs/router_dropout_0_seed123.yaml

python train.py --config configs/router_dropout_03_seed123.yaml
python eval.py --config configs/router_dropout_03_seed123.yaml
```

生成汇总表格：

```powershell
python scripts/summarize_dropout_ablation.py
```
