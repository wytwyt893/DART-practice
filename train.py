import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataloader import SyntheticDataset
from models.router import MLPRouter

from pathlib import Path

from utils.config import load_config #导入utils/config.py中的load_config函数，用于从 YAML 文件中读取实验配置，返回一个包含配置信息的字典对象，供后续训练流程使用
from utils.seed import set_seed #导入utils/seed.py中的set_seed函数，用于设置随机种子，确保训练过程的可重复性，减少每次结果的波动
from utils.metrics import evaluate #导入utils/metrics.py中的evaluate函数，用于在验证集上评估模型的性能，返回平均损失和准确率，供训练循环中每个 epoch 后的评估使用

#==============================训练流程的最小闭环======================================
def main() -> None:
    """
    构建一个最小可运行的训练闭环：
    1. 生成 synthetic 数据
    2. 划分训练集 / 验证集
    3. 构建一个简单的 MLP baseline
    4. 训练并输出每个 epoch 的结果
    """
    config = load_config("configs/router_sanity.yaml") # 从指定路径加载 YAML 配置文件，返回一个包含配置信息的字典对象，供后续训练流程使用
    print(f"Loaded config: {config['experiment_name']}")

    # 1. 选择固定种子，固定随机性，减少每次结果波动。
    seed = config.get("seed", 42)
    set_seed(seed)

    # 2.选择CPU/GPU,如果有 GPU 就优先使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 构造 toy 数据集。
    # num_samples=1200: 一共 1200 个样本
    # feature_dim=10112: 每个样本是 10112 维特征向量
    # num_routes=3: 三分类任务
    dataset = SyntheticDataset(num_samples=config["data"]["num_samples"], feature_dim=config["data"]["feature_dim"], num_routes=config["data"]["num_routes"])

    # 4. 按 8:2 划分训练集和验证集。
    train_ratio = config["data"]["train_ratio"]
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader 负责把单个样本打包成 batch。
    # 训练集打乱顺序，避免顺序偏差影响训练。
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["train_batch_size"], shuffle=True)

    # 验证集不需要打乱，只负责稳定评估。
    # batch_size 可以稍大一些，因为验证时没有反向传播。
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["val_batch_size"], shuffle=False)

    # 5.创建最小 baseline 模型
    # input_dim 必须与 feature_dim 对齐，否则第一层接不上输入。
    model = MLPRouter(
        input_dim=config["model"]["input_dim"], 
        hidden_dim=config["model"]["hidden_dim"], 
        num_routes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # 6.定义 Adam 优化器，适合快速起一个 baseline。
    lr = config["training"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7.这里是标准多分类损失函数。
    loss_fn = nn.CrossEntropyLoss()

    log_path = Path(config["outputs"]["log_file"]) # 定义日志文件路径，存储训练过程中的指标数据
    log_path.parent.mkdir(exist_ok=True, parents=True) # 确保 logs 目录存在，如果不存在就创建它, parents=True 允许创建多层目录，exist_ok=True 则在目录已经存在时不会报错
    
    last_ckpt_path = Path(config["outputs"]["last_checkpoint"]) # 定义最后一次训练的模型检查点文件路径，保存训练结束时的模型权重
    best_ckpt_path = Path(config["outputs"]["best_checkpoint"]) # 定义最佳模型检查点文件路径，保存验证集上表现最好的模型权重

    last_ckpt_path.parent.mkdir(exist_ok=True, parents=True)
    best_ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    best_val_acc = 0.0 # 初始化最佳验证准确率，用于后续比较和更新最佳模型检查点

    # 打开日志文件，准备写入训练过程中的指标数据
    # 当前 train.py 的日志文件每次运行会覆盖旧结果，因为用了 "w" 模式
    with open(log_path, "w", encoding="utf-8") as log_file: 
        # 8. 训练循环
        # 先训练 10 个 epoch，作为 sanity check。
        epochs = config["training"]["epochs"]

        for epoch in range(1, epochs + 1):
            # 切换到训练模式。
            model.train()

            total_loss = 0.0 #初始化累计损失和正确数
            total_correct = 0
            total_examples = 0

            for features, route_labels in train_loader:
                # 当前 batch 放到设备上。
                features = features.to(device)
                route_labels = route_labels.to(device)

                # 清空上一步累积的梯度。
                optimizer.zero_grad()

                # 前向传播，得到当前 batch 的预测 logits。
                route_logits = model(features)

                # 计算当前 batch 的损失。
                loss = loss_fn(route_logits, route_labels)

                # 反向传播，计算每个参数的梯度。
                loss.backward()

                # 根据梯度更新参数。
                optimizer.step()

                # 累计当前 epoch 的损失和正确数。
                total_loss += loss.item() * route_labels.size(0)
                total_correct += (route_logits.argmax(dim=1) == route_labels).sum().item()
                total_examples += route_labels.size(0)

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
