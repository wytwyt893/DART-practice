from torch import nn
from torch.utils.data import DataLoader
import torch

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
        for features, route_labels in dataloader:
            # 把当前 batch 放到同一个设备上，避免设备不一致报错。
            features = features.to(device)
            route_labels = route_labels.to(device)

            # 前向传播：输入特征，输出每个类别的 logits。
            route_logits = model(features)

            # 计算当前 batch 的损失。
            loss = loss_fn(route_logits, route_labels)

            # 这里做的是“按样本数加权”的累计。
            # 因为 loss.item() 是当前 batch 的平均 loss，
            # 乘上 route_labels.size(0) 才能还原成当前 batch 的总 loss。
            total_loss += loss.item() * route_labels.size(0)

            # 统计当前 batch 预测正确了多少个样本。
            total_correct += (route_logits.argmax(dim=1) == route_labels).sum().item()

            # 统计一共看了多少个样本。
            total_examples += route_labels.size(0)

    # 把累计值转换成整个验证集上的平均指标。
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy
