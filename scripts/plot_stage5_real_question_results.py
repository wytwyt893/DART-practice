from pathlib import Path
import re

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXPERIMENTS = {
    "Paper-dim mock": {
        "log_path": PROJECT_ROOT / "logs" / "router_real_format_paper_dims_mock_log.txt",
        "description": "Text/Image/Question all mock",
        "question_feature": "Mock + signal",
    },
    "Real question embedding": {
        "log_path": PROJECT_ROOT / "logs" / "router_real_format_real_question_mock_log.txt",
        "description": "Text/Image mock, Question real MiniLM",
        "question_feature": "MiniLM [384]",
    },
}


LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+"
    r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+"
    r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


def parse_log(log_path: Path) -> dict[str, list[float]]:
    """
    从训练日志中解析每个 epoch 的 train/val loss 和 accuracy。

    日志格式来自 train_real_format_mock.py，例如：
    Epoch 01 | train_loss=0.7986 train_acc=0.7430 | val_loss=0.3792 val_acc=0.9667
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if match is None:
            continue

        epoch, train_loss, train_acc, val_loss, val_acc = match.groups()
        metrics["epoch"].append(int(epoch))
        metrics["train_loss"].append(float(train_loss))
        metrics["train_acc"].append(float(train_acc))
        metrics["val_loss"].append(float(val_loss))
        metrics["val_acc"].append(float(val_acc))

    if not metrics["epoch"]:
        raise ValueError(f"No valid epoch metrics found in: {log_path}")

    return metrics


def best_epoch(metrics: dict[str, list[float]]) -> tuple[int, float, float]:
    """返回验证集准确率最高的 epoch、best val_acc 和对应 val_loss。"""
    best_index = max(range(len(metrics["val_acc"])), key=lambda i: metrics["val_acc"][i])
    return (
        metrics["epoch"][best_index],
        metrics["val_acc"][best_index],
        metrics["val_loss"][best_index],
    )


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage5_real_question_comparison.png"

    parsed = {
        name: {
            **info,
            "metrics": parse_log(info["log_path"]),
        }
        for name, info in EXPERIMENTS.items()
    }

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.1], hspace=0.35, wspace=0.25)

    ax_acc = fig.add_subplot(grid[0, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    colors = {
        "Paper-dim mock": "#2E6F9E",
        "Real question embedding": "#C75146",
    }

    for name, info in parsed.items():
        metrics = info["metrics"]
        ax_acc.plot(
            metrics["epoch"],
            metrics["val_acc"],
            marker="o",
            linewidth=2.4,
            label=name,
            color=colors[name],
        )
        ax_loss.plot(
            metrics["epoch"],
            metrics["val_loss"],
            marker="o",
            linewidth=2.4,
            label=name,
            color=colors[name],
        )

    ax_acc.set_title("Validation Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Acc")
    ax_acc.set_ylim(0.75, 1.02)
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend()

    ax_loss.set_title("Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()

    rows = []
    for name, info in parsed.items():
        epoch, val_acc, val_loss = best_epoch(info["metrics"])
        rows.append(
            [
                name,
                info["description"],
                info["question_feature"],
                f"{epoch}",
                f"{val_acc:.4f}",
                f"{val_loss:.4f}",
            ]
        )

    ax_table.axis("off")
    table = ax_table.table(
        cellText=rows,
        colLabels=[
            "Experiment",
            "Feature Setting",
            "Question Feature",
            "Best Epoch",
            "Best Val Acc",
            "Best Val Loss",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#EAE3D2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FBFAF6")
        cell.set_edgecolor("#B8B0A0")

    fig.suptitle(
        "Stage 5.1: Real Question Embedding vs Mock Question Feature",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "Note: lower accuracy after real question embedding is expected because the artificial mock signal in question_feature is removed.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
