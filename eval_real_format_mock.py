from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.real_format_dataloader import RealFormatMockFeatureDataset, real_format_collate_fn
from models.mini_dynamic_selector import MiniDynamicSelector
from train_real_format_mock import evaluate_multisource
from utils.config import load_config
from utils.seed import set_seed

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_real_format_mock.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()

def main() -> None:
    """
    Stage 4 的独立评估脚本。

    它和 train_real_format_mock.py 的关系可以这样理解：
    - train_real_format_mock.py: 负责训练模型，并保存 last/best checkpoint。
    - eval_real_format_mock.py: 不训练，只加载 best checkpoint，在验证集上重新评估。

    单独写 eval 文件不是为了提高准确率，而是为了模拟正式科研实验流程：
    训练结束后，必须能脱离训练脚本，独立复查某个 checkpoint 的效果。
    """
    # 1. 读取和训练阶段完全相同的配置文件。
    #
    # eval 必须使用同一个 config，否则可能出现三类问题：
    # - Dataset 读取的数据范围不同；
    # - 模型维度和 checkpoint 里的参数形状对不上；
    # - best checkpoint 的路径找错。
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    print(f"Loaded config: {config['experiment_name']}")

    # 2. 固定随机种子。
    #
    # 这里尤其重要，因为我们会重新调用 random_split。
    # 只有 seed 和 train_ratio 与训练阶段一致，eval 才能切到同一份验证集。
    set_seed(config["seed"])

    # 3. 选择运行设备。
    #
    # 有 CUDA 就用 GPU；没有 CUDA 就退回 CPU。
    # checkpoint 加载时也会通过 map_location 映射到这个 device。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 重建真实格式 Dataset。
    #
    # 注意：eval 不会从 checkpoint 里读取数据。
    # checkpoint 保存的是模型参数，不保存 Dataset。
    # 所以 eval 阶段必须按照训练时的 config 重新构造同一套数据。
    dataset = RealFormatMockFeatureDataset(
        data_path=config["data"]["data_path"],
        max_samples=config["data"]["max_samples"],
        samples_per_category=config["data"].get("samples_per_category"),
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
        use_real_question_embedding=config["data"].get("use_real_question_embedding", False),
        question_embed_model_id=config["data"].get(
            "question_embed_model_id",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )

    # 5. 按照训练阶段相同的 train_ratio 重新切分 train/val。
    #
    # eval 只需要 val_dataset，所以 train_dataset 用 _ 接住。
    # 只要 seed 一样，random_split 的结果就会和训练阶段保持一致。
    train_size = int(len(dataset) * config["data"]["train_ratio"])
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    # 6. 创建验证集 DataLoader。
    #
    # 这里必须传入 real_format_collate_fn。
    # 原因是官方真实格式里的 table/question/answer/image 等字段不是定长 tensor，
    # PyTorch 默认 collate 无法直接把它们堆叠成 batch。
    # real_format_collate_fn 会把 tensor 特征 stack，把原始字段保留成 list。
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
        collate_fn=real_format_collate_fn,
    )

    # 7. 重建和训练阶段完全相同结构的模型。
    #
    # checkpoint 只保存 model_state_dict，也就是参数值；
    # 它不保存 MiniDynamicSelector 这个 Python 类本身。
    # 因此加载参数之前，必须先用相同维度重新创建模型对象。
    model = MiniDynamicSelector(
        text_dim=config["model"]["text_dim"],
        image_dim=config["model"]["image_dim"],
        question_dim=config["model"]["question_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_routes=config["model"]["num_routes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # 8. 找到训练阶段保存的 best checkpoint。
    #
    # best checkpoint 指的是训练过程中验证集准确率最高的那一次模型快照。
    checkpoint_path = Path(config["outputs"]["best_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}. "
            "Please run train_real_format_mock.py first."
        )

    # 9. 加载 checkpoint。
    #
    # weights_only=True 更安全，适合加载自己训练出来的权重字典。
    # map_location=device 的作用是：无论 checkpoint 当时在哪个设备保存，
    # 现在都加载到当前脚本选择的 CPU/GPU 上。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 10. 把 checkpoint 里的模型参数装回当前 model。
    #
    # 这一步要求当前 model 的结构和训练时完全一致。
    # 如果维度不一致，比如 hidden_dim 改了，这里会报 size mismatch。
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val_acc: {checkpoint['val_acc']}")

    # 11. 独立评估。
    #
    # 这里复用 train_real_format_mock.py 里的 evaluate_multisource。
    # 它内部会执行：
    # model.eval()
    # torch.no_grad()
    # 前向传播
    # 计算 CrossEntropyLoss 和 accuracy
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate_multisource(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"独立评估结果 | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
