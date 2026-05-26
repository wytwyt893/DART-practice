import argparse
import re
from pathlib import Path

#==================================训练曲线绘制脚本====================================

# 1. 这个脚本的作用是从 train.py 的日志里提取训练曲线数据，手动写死的一组训练结果
DEFAULT_HISTORY = [
    {"epoch": 1, "train_loss": 0.5090, "train_acc": 0.7458, "val_loss": 0.3351, "val_acc": 0.8708},
    {"epoch": 2, "train_loss": 0.0815, "train_acc": 0.9990, "val_loss": 0.1746, "val_acc": 0.9792},
    {"epoch": 3, "train_loss": 0.0196, "train_acc": 1.0000, "val_loss": 0.1475, "val_acc": 0.9750},
    {"epoch": 4, "train_loss": 0.0079, "train_acc": 1.0000, "val_loss": 0.1400, "val_acc": 0.9750},
    {"epoch": 5, "train_loss": 0.0048, "train_acc": 1.0000, "val_loss": 0.1333, "val_acc": 0.9708},
    {"epoch": 6, "train_loss": 0.0034, "train_acc": 1.0000, "val_loss": 0.1280, "val_acc": 0.9708},
    {"epoch": 7, "train_loss": 0.0026, "train_acc": 1.0000, "val_loss": 0.1246, "val_acc": 0.9708},
    {"epoch": 8, "train_loss": 0.0021, "train_acc": 1.0000, "val_loss": 0.1216, "val_acc": 0.9708},
    {"epoch": 9, "train_loss": 0.0017, "train_acc": 1.0000, "val_loss": 0.1199, "val_acc": 0.9708},
    {"epoch": 10, "train_loss": 0.0015, "train_acc": 1.0000, "val_loss": 0.1182, "val_acc": 0.9708},
]

# 2. 定义一个正则表达式，用来从日志文本里提取每一行的训练记录。
LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)

# 3. 解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training curves for the current sanity-check experiment."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional text log to parse. If omitted, the script uses the current default run history.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs") / "training_curves.png",
        help="Output image path.",
    )
    return parser.parse_args()

# 4. 从日志文件里加载训练历史数据，如果没有提供日志文件，就用默认的那组数据。
def load_history(log_file: Path | None) -> list[dict]:
    if log_file is None:
        return DEFAULT_HISTORY

    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    history = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if match is None:
            continue
        history.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "train_acc": float(match.group(3)),
                "val_loss": float(match.group(4)),
                "val_acc": float(match.group(5)),
            }
        )

    if not history:
        raise ValueError("No training records matched the expected log format.")
    return history

# 5. 根据加载的历史数据绘制训练曲线，并保存成图片。
def plot_history(history: list[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it in the current environment first."
        ) from exc

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Synthetic Sanity Check Training Curves", fontsize=14)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2, label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, label="Train Acc")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=2, label="Val Acc")
    axes[1].axhline(1 / 3, color="gray", linestyle="--", linewidth=1.5, label="Random Guess (33%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

# 6. 主函数，负责整体流程控制：解析参数、加载历史数据、绘制并保存图像。
def main() -> None:
    args = parse_args()
    history = load_history(args.log_file)
    plot_history(history, args.out)


if __name__ == "__main__":
    main()
