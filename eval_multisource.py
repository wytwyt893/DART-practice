from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.dataloader import MultiSourceSyntheticDataset
from models.mini_dynamic_selector import MiniDynamicSelector
from train_multisource import evaluate_multisource
from utils.config import load_config
from utils.seed import set_seed

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_multisource_sanity.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()

def main() -> None:
    # 1. 加载和训练阶段完全相同的配置文件。
    # eval 必须使用同一个 config，否则数据维度、模型结构或 checkpoint 路径可能对不上。
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    print(f"Loaded config: {config['experiment_name']}")

    # 2. 设置随机种子。
    # 这里非常重要：train_multisource.py 里用 random_split 划分 train/val；
    # eval 阶段要重新构造同一个 dataset 并得到同一个 val split，
    # 所以必须使用同一个 seed，才能保证验证集和训练时的验证集一致。
    set_seed(config["seed"])

    # 3. 重建 multi-source synthetic dataset。
    # 注意：eval 不会直接从 checkpoint 里读取数据，只读取模型参数；
    # 数据集仍然要按训练时的配置重新构造。
    dataset = MultiSourceSyntheticDataset(
        num_samples=config["data"]["num_samples"],
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
    )

    # 4. 按照训练阶段相同的 train_ratio 重新切分数据。
    # 这里只需要 val_dataset，所以 train_dataset 用 _ 接住即可。
    train_size = int(len(dataset) * config["data"]["train_ratio"])
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
    )

    # 5. 创建和训练阶段完全相同结构的 MiniDynamicSelector。
    # checkpoint 只保存参数，不保存模型结构；
    # 所以加载 checkpoint 前，必须先用同样的维度重新创建模型对象。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniDynamicSelector(
        text_dim=config["model"]["text_dim"],
        image_dim=config["model"]["image_dim"],
        question_dim=config["model"]["question_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_routes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # 6. 加载训练阶段保存的 best checkpoint。
    checkpoint_path = Path(config["outputs"]["best_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. "
            "Please run train_multisource.py first."
        )

    # weights_only=True 在新版本 PyTorch 中更安全；
    # 如果你的 PyTorch 版本不支持该参数，可以改回 torch.load(checkpoint_path, map_location=device)。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 7. 把 checkpoint 里的模型参数装回当前 model。
    # model_state_dict 对应 train_multisource.py 里保存的 model.state_dict()。
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val_acc: {checkpoint['val_acc']}")

    # 8. 调用和训练阶段相同的 evaluate_multisource。
    # 这里暂时从 train_multisource.py 复用该函数；
    # 后续重构时可以把 evaluate_multisource 移到 utils/metrics.py。
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_multisource(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"Eval result | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
