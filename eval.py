import yaml 
import torch
from pathlib import Path
from models.router import SimpleRouter
from torch import nn
from torch.utils.data import DataLoader, random_split
from data.dataloader import SyntheticDataset
import random

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_config(config_path: str) -> dict:
    """
    从 YAML 文件中读取实验配置。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) # yaml.safe_load() 是 PyYAML 库提供的一个函数，用于从 YAML 格式的字符串或文件中解析数据，并将其转换为 Python 对象（通常是字典）。相比于 yaml.load()，yaml.safe_load() 在解析过程中会限制一些不安全的 YAML 标签，避免执行潜在的恶意代码，因此更推荐使用 safe_load 来加载配置文件。
    return config

def evaluate(model, dataloader, device):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            loss = loss_fn(logits, labels)

            total_loss += loss.item() * features.size(0)

            total_correct += (logits.argmax(dim=1) == labels).sum().item()

            total_examples += features.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def main():

    # 1.读取 config
    config = load_config("configs/router_sanity.yaml") # 从指定路径加载配置文件，获取实验的各种参数设置
    
    # 2.打印 best checkpoint 路径
    best_ckpt_path = Path(config["outputs"]["best_checkpoint"]) # 从配置文件中获取最佳模型检查点的路径，并将其转换为 Path 对象，方便后续文件操作
    print(best_ckpt_path)
    print(best_ckpt_path.exists())

    # 3.设置随机种子
    set_seed(config.get("seed", 42))

    # 4.设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5.重建dataset
    dataset = SyntheticDataset(
        num_samples=config["data"]["num_samples"],
        feature_dim=config["data"]["feature_dim"],
        num_classes=config["data"]["num_routes"],
    )
    
    # 6.划分训练集和验证集
    train_ratio = config["data"]["train_ratio"]
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 7.构建验证集 dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
    )

    # 8.加载 best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)

    # 9.打印 checkpoint 里的内容，确认我们能正确加载它
    print(checkpoint.keys())
    print("Best checkpoint epoch:", checkpoint["epoch"])
    print("Best checkpoint val_acc:", checkpoint["val_acc"])

    # 10.初始化模型
    model = SimpleRouter(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # 11.加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\nLoaded model weights successfully.")

    # 12.评估模型在验证集上的表现，输出 loss 和 accuracy
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Eval result | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()

