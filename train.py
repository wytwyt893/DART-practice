import random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataloader import SyntheticDataset
from models.router import SimpleRouter

from pathlib import Path

#==============================训练流程的最小闭环======================================

def set_seed(seed: int = 42) -> None:
    """
    固定随机种子，尽量让每次运行结果保持一致，方便复现实验。

    Args:
        seed: 同时作用于 Python random 和 PyTorch 的随机种子。
    """
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    在验证集上完整跑一轮，返回平均 loss 和准确率。

    Args:
        model: 要评估的模型。
        dataloader: 验证集的 DataLoader，每次提供一个 batch。
        device: 当前运行设备，通常是 CPU 或 GPU。

    Returns:
        (avg_loss, accuracy)
    """
    # 切换到评估模式。
    # 如果以后模型中加入 Dropout / BatchNorm，这一步会影响它们的行为。
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # 验证阶段不需要反向传播，因此关闭梯度计算以节省显存和时间。
    with torch.no_grad():
        for features, labels in dataloader:
            # 把当前 batch 放到同一个设备上，避免设备不一致报错。
            features = features.to(device)
            labels = labels.to(device)

            # 前向传播：输入特征，输出每个类别的 logits。
            logits = model(features)

            # 计算当前 batch 的损失。
            loss = loss_fn(logits, labels)

            # 这里做的是“按样本数加权”的累计。
            # 因为 loss.item() 是当前 batch 的平均 loss，
            # 乘上 labels.size(0) 才能还原成当前 batch 的总 loss。
            total_loss += loss.item() * labels.size(0)

            # 统计当前 batch 预测正确了多少个样本。
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

            # 统计一共看了多少个样本。
            total_examples += labels.size(0)

    # 把累计值转换成整个验证集上的平均指标。
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def main() -> None:
    """
    构建一个最小可运行的训练闭环：
    1. 生成 synthetic 数据
    2. 划分训练集 / 验证集
    3. 构建一个简单的 MLP baseline
    4. 训练并输出每个 epoch 的结果
    """
    # 1. 选择固定种子，固定随机性，减少每次结果波动。
    set_seed()

    # 2.选择CPU/GPU,如果有 GPU 就优先使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 构造 toy 数据集。
    # num_samples=1200: 一共 1200 个样本
    # feature_dim=10112: 每个样本是 10112 维特征向量
    # num_classes=3: 三分类任务
    dataset = SyntheticDataset(num_samples=1200, feature_dim=10112, num_classes=3)

    # 4. 按 8:2 划分训练集和验证集。
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader 负责把单个样本打包成 batch。
    # 训练集打乱顺序，避免顺序偏差影响训练。
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 验证集不需要打乱，只负责稳定评估。
    # batch_size 可以稍大一些，因为验证时没有反向传播。
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 5.创建最小 baseline 模型
    # input_dim 必须与 feature_dim 对齐，否则第一层接不上输入。
    model = SimpleRouter(input_dim=10112, hidden_dim=256, num_classes=3).to(device)

    # 6.定义 Adam 优化器，适合快速起一个 baseline。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 7.这里是标准多分类损失函数。
    loss_fn = nn.CrossEntropyLoss()

    log_path = Path("logs") / "router_sanity_log.txt" # 定义日志文件路径，存储训练过程中的指标数据
    log_path.parent.mkdir(exist_ok=True, parents=True) # 确保 logs 目录存在，如果不存在就创建它, parents=True 允许创建多层目录，exist_ok=True 则在目录已经存在时不会报错
    
    checkpoint_dir = Path("checkpoints") # 定义模型检查点目录路径，存储训练过程中保存的模型权重文件
    checkpoint_dir.mkdir(exist_ok=True, parents=True) # 确保 checkpoints 目录存在，如果不存在就创建它, parents=True 允许创建多层目录，exist_ok=True 则在目录已经存在时不会报错

    last_ckpt_path = checkpoint_dir / "router_sanity_last.pth" # 定义最后一次训练的模型检查点文件路径，保存训练结束时的模型权重
    best_ckpt_path = checkpoint_dir / "router_sanity_best.pth" # 定义最佳模型检查点文件路径，保存验证集上表现最好的模型权重

    best_val_acc = 0.0 # 初始化最佳验证准确率，用于后续比较和更新最佳模型检查点

    # 打开日志文件，准备写入训练过程中的指标数据
    # 当前 train.py 的日志文件每次运行会覆盖旧结果，因为用了 "w" 模式
    with open(log_path, "w", encoding="utf-8") as log_file: 
        # 8. 训练循环
        # 先训练 10 个 epoch，作为 sanity check。
        epochs = 10
        for epoch in range(1, epochs + 1):
            # 切换到训练模式。
            model.train()

            total_loss = 0.0 #初始化累计损失和正确数
            total_correct = 0
            total_examples = 0

            for features, labels in train_loader:
                # 当前 batch 放到设备上。
                features = features.to(device)
                labels = labels.to(device)

                # 清空上一步累积的梯度。
                optimizer.zero_grad()

                # 前向传播，得到当前 batch 的预测 logits。
                logits = model(features)

                # 计算当前 batch 的损失。
                loss = loss_fn(logits, labels)

                # 反向传播，计算每个参数的梯度。
                loss.backward()

                # 根据梯度更新参数。
                optimizer.step()

                # 累计当前 epoch 的损失和正确数。
                total_loss += loss.item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_examples += labels.size(0)

            # 计算当前 epoch 的训练集平均指标。
            train_loss = total_loss / total_examples
            train_acc = total_correct / total_examples

            # 9.每个 epoch 后在验证集评估
            # 在验证集上做一次完整评估。
            val_loss, val_acc = evaluate(model, val_loader, device)
            
            # 构造当前 epoch 的模型检查点数据，包含 epoch 编号、模型权重、优化器状态、验证集指标等信息，方便后续保存和加载模型
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, last_ckpt_path) # 保存当前 epoch 的模型检查点到 last_ckpt_path 定义的路径，覆盖之前的检查点文件
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, best_ckpt_path) # 如果当前 epoch 的验证准确率超过之前的最佳值，就更新最佳模型检查点文件，保存当前 epoch 的模型权重和相关信息

            # 10.打印这一轮的训练 / 验证结果。
            # 同时把结果写到日志文件里，方便后续查看。
            log_line = f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            print(log_line)
            log_file.write(log_line + "\n") # 把每个 epoch 的结果写到日志文件里，方便后续查看和分析

    # 正常训练结束后的提示。
    print(f"Finished training on {device}.")


if __name__ == "__main__":
    main()
