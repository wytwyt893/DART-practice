from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.dataloader import MultiSourceSyntheticDataset
from models.mini_dynamic_selector import MiniDynamicSelector
from utils.config import load_config
from utils.seed import set_seed


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


def main() -> None:
    # 1. 加载配置文件，获取训练参数和数据参数。
    config_path = "configs/router_multisource_sanity.yaml"
    config = load_config(config_path)
    print(f"Loaded config: {config['experiment_name']}")

    # 2. 设置随机种子，确保结果可复现。
    set_seed(config["seed"])

    # 3. 创建数据集和数据加载器。
    dataset = MultiSourceSyntheticDataset(
        num_samples=config["data"]["num_samples"],
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
    )

    # 4. 划分训练集和验证集，创建对应的 DataLoader。
    train_size = int(len(dataset) * config["data"]["train_ratio"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
    )


    # 先打印一个 batch 的 shape，作为 multi-source 数据流健康检查。
    first_batch = next(iter(train_loader))
    print(f"Text feature shape: {first_batch['text_feature'].shape}")
    print(f"Image feature shape: {first_batch['image_feature'].shape}")
    print(f"Question feature shape: {first_batch['question_feature'].shape}")
    print(f"Route label shape: {first_batch['route_label'].shape}")

    # 5. 创建模型，定义损失函数和优化器。
    model = MiniDynamicSelector(
        text_dim=config["model"]["text_dim"],
        image_dim=config["model"]["image_dim"],
        question_dim=config["model"]["question_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_routes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"],
    )

    # 6. 选择设备（GPU 或 CPU），并把模型移动到设备上。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器。这里使用交叉熵损失和 Adam 优化器。
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # 7. 训练循环，包含每个 epoch 的训练和验证，以及日志记录和模型保存。
    log_path = Path(config["outputs"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    best_val_acc = 0.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        # 训练一个 epoch，并在验证集上评估模型性能。
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        # 评估模型在验证集上的表现，计算平均损失和准确率。
        val_loss, val_acc = evaluate_multisource(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        # 打印日志，并把训练和验证指标写入日志文件。
        log_line = (
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(log_line)


        with log_path.open("a", encoding="utf-8") as file:
            file.write(log_line + "\n")

        # 每个 epoch 结束后保存一次模型检查点，并在验证准确率提升时更新最佳模型检查点。
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

    print(f"Finished multi-source training on {device}.")


if __name__ == "__main__":
    main()
