import random

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataloader import SyntheticDataset
from models.router import SimpleRouter

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

    # 8. 训练循环
    # 先训练 10 个 epoch，作为 sanity check。
    epochs = 10
    for epoch in range(1, epochs + 1):
        # 切换到训练模式。
        model.train()

        total_loss = 0.0
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

        # 10.打印这一轮的训练 / 验证结果。
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # 正常训练结束后的提示。
    print(f"Finished training on {device}.")


if __name__ == "__main__":
    main()
