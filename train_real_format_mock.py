from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.real_format_dataloader import RealFormatMockFeatureDataset, real_format_collate_fn
from models.mini_dynamic_selector import MiniDynamicSelector
from utils.config import load_config
from utils.seed import set_seed

import argparse

# 这个函数的作用是把一个 batch 的数据从 CPU 移动到 GPU（如果可用的话）
# 以便模型训练和评估时能够在 GPU 上进行计算
def move_batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    """
    multi-source 版本和旧 train.py 最大的不同点：
    旧版 batch 是 (features, labels)，这里 batch 是 dict。

    这个函数专门负责把 dict batch 里的四个 tensor 取出来并移动到 GPU/CPU。
    """
    text_feature = batch["text_feature"].to(device)
    image_feature = batch["image_feature"].to(device)
    question_feature = batch["question_feature"].to(device)
    route_label = batch["route_label"].to(device)
    return text_feature, image_feature, question_feature, route_label

# 这个函数的作用是在验证集上评估模型的性能
# 计算平均损失和准确率
# 供训练循环中每个 epoch 后的评估使用
def evaluate_multisource( #作用是在验证集上评估模型的性能，计算平均损失和准确率，供训练循环中每个 epoch 后的评估使用。
    model: MiniDynamicSelector,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证流程：和训练流程很像，但不反向传播，也不更新参数。"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            text_feature, image_feature, question_feature, route_label = move_batch_to_device(
                batch, device
            )

            route_logits = model(text_feature, image_feature, question_feature)
            loss = loss_fn(route_logits, route_label)

            batch_size = route_label.size(0)
            total_loss += loss.item() * batch_size

            predictions = route_logits.argmax(dim=1)
            correct += (predictions == route_label).sum().item()
            total += batch_size

    return total_loss / total, correct / total

# 这个函数的作用是训练一个 epoch 的模型，并返回平均损失和准确率。
def train_one_epoch(
    model: MiniDynamicSelector,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch：这是旧 train.py 里的标准训练套路。"""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        text_feature, image_feature, question_feature, route_label = move_batch_to_device(
            batch, device
        )

        # 重点：这里是 multi-source 版本的“套娃调用链”。
        #
        # 1. model 是 MiniDynamicSelector 对象。
        # 2. model(text_feature, image_feature, question_feature)
        #    会调用 models/mini_dynamic_selector.py 里的 MiniDynamicSelector.forward。
        # 3. MiniDynamicSelector.forward 内部先 torch.cat 三路 feature。
        # 4. 然后调用 self.router(combined_feature)。
        # 5. self.router 是 models/router.py 里的 MLPRouter 对象，
        #    所以最后会调用 MLPRouter.forward，把 combined_feature 送进 MLP。
        route_logits = model(text_feature, image_feature, question_feature)

        # CrossEntropyLoss 的输入格式：
        # route_logits: [batch_size, num_routes]
        # route_label:  [batch_size]
        loss = loss_fn(route_logits, route_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = route_label.size(0)
        total_loss += loss.item() * batch_size

        predictions = route_logits.argmax(dim=1)
        correct += (predictions == route_label).sum().item()
        total += batch_size

    return total_loss / total, correct / total

# 这个函数的作用是保存训练过程中的模型快照（checkpoint）
# 包括当前 epoch、模型参数、优化器状态、验证损失和验证准确率等信息
# 以便后续恢复训练或者分析训练过程。
def save_checkpoint(
    path: str,
    epoch: int,
    model: MiniDynamicSelector,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    val_acc: float,
) -> None:
    """保存训练快照。这里保存的是模型参数和训练状态，不保存整个模型对象。"""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save( # 这里保存的内容是一个 dict，包含了 epoch、模型参数、优化器状态、验证损失和验证准确率等信息，可以用于后续恢复训练或者分析训练过程。
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        checkpoint_path,
    )

# parse_args函数作用是解析命令行参数
# 特别是用于指定配置文件的路径。
# 它使用 argparse 库来定义和处理命令行参数
# 使得用户可以通过命令行指定不同的配置文件来运行训练脚本
# 从而灵活地调整训练设置
def parse_args():
    parser = argparse.ArgumentParser() #创建一个 ArgumentParser 对象，用于处理命令行参数
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_real_format_mock.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()

def main() -> None:
    # 1. 读取 Stage 4 专用配置。
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    print(f"Loaded config: {config['experiment_name']}")
    
    # 2. 设置随机种子。
    set_seed(config["seed"])

    # 3. 选择设备（GPU 或 CPU）。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 创建数据集和数据加载器。
    dataset = RealFormatMockFeatureDataset(
        data_path=config["data"]["data_path"], #数据文件路径
        max_samples=config["data"]["max_samples"], #最大样本数
        samples_per_category=config["data"].get("samples_per_category"), #每个类别的样本数（可选）
        text_dim=config["data"]["text_dim"], 
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"], 

        use_real_question_embedding=config["data"].get("use_real_question_embedding", False), #是否使用真实问题嵌入，False 表示使用随机特征
        question_embed_model_id=config["data"].get( #生成问题嵌入使用的模型 ID，默认为 "sentence-transformers/all-MiniLM-L6-v2"
            "question_embed_model_id",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),

        #新增的两个参数专门用于控制 question_feature 的生成方式。
        # 把 YAML里的缓存开关传给 Dataset
        use_cached_question_features=config["data"].get("use_cached_question_features", False),
        question_feature_path=config["data"].get("question_feature_path"),
    )

    # 5. 划分训练集和验证集，创建对应的 DataLoader。
    train_size = int(len(dataset) * config["data"]["train_ratio"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=real_format_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
        collate_fn=real_format_collate_fn,
    )

    
    first_batch = next(iter(train_loader))

    print("训练前 batch 检查:")
    print(f"  text_feature: {first_batch['text_feature'].shape}")
    print(f"  image_feature: {first_batch['image_feature'].shape}")
    print(f"  question_feature: {first_batch['question_feature'].shape}")
    print(f"  route_label: {first_batch['route_label'].shape}")
    print(f"  category 示例: {first_batch['category'][:5]}")
    print(f"  question_id 示例: {first_batch['question_id'][:5]}")

    # 6. 创建模型、损失函数和优化器。
    model = MiniDynamicSelector(
        text_dim=config["model"]["text_dim"],
        image_dim=config["model"]["image_dim"],
        question_dim=config["model"]["question_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_routes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # 7. 训练循环：这里是标准的训练流程，每个 epoch 包含一个训练阶段和一个验证阶段，最后保存模型快照。
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    log_path = Path(config["outputs"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    best_val_acc = 0.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch( #调用上面定义的 train_one_epoch 函数，进行一个 epoch 的训练，并返回平均损失和准确率。
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_multisource( #调用上面定义的 evaluate_multisource 函数，在验证集上评估模型性能，计算平均损失和准确率。
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        log_line = (
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(log_line)

        with log_path.open("a", encoding="utf-8") as file:
            file.write(log_line + "\n")

        save_checkpoint(
            path=config["outputs"]["last_checkpoint"],
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=config["outputs"]["best_checkpoint"],
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                val_acc=val_acc,
            )

    print(f"真实格式 mock 训练完成，运行设备: {device}")

if __name__ == "__main__":
    main()