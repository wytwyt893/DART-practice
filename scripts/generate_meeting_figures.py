import argparse
import re
from collections import defaultdict
from pathlib import Path


# 这个脚本用于生成组会汇报图。
# 它不会重新训练模型，只会读取 logs/*.txt 中已经保存好的训练记录，
# 然后生成 dropout 消融图、训练曲线图、实验表格图和工程结构示意图。


LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


# 当前已经完成的 dropout ablation 实验清单。
# 如果后续补跑 seed=123/dropout=0.1，直接在这里新增一行即可。
EXPERIMENTS = [
    {
        "name": "dropout_0_seed42",
        "dropout": 0.0,
        "seed": 42,
        "log_path": Path("logs/router_dropout_0_log.txt"),
    },
    {
        "name": "dropout_01_seed42",
        "dropout": 0.1,
        "seed": 42,
        "log_path": Path("logs/router_sanity_log.txt"),
    },
    {
        "name": "dropout_03_seed42",
        "dropout": 0.3,
        "seed": 42,
        "log_path": Path("logs/router_dropout_03_log.txt"),
    },
    {
        "name": "dropout_0_seed123",
        "dropout": 0.0,
        "seed": 123,
        "log_path": Path("logs/router_dropout_0_seed123_log.txt"),
    },
    {
        "name": "dropout_03_seed123",
        "dropout": 0.3,
        "seed": 123,
        "log_path": Path("logs/router_dropout_03_seed123_log.txt"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate meeting-ready figures for the mini TableDART router project."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def parse_log(log_path: Path) -> list[dict]:
    """把单个训练日志解析成 epoch 记录列表。"""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if match is None:
            continue

        records.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "train_acc": float(match.group(3)),
                "val_loss": float(match.group(4)),
                "val_acc": float(match.group(5)),
            }
        )

    if not records:
        raise ValueError(f"No epoch records found in {log_path}")
    return records


def load_experiments() -> list[dict]:
    """读取所有实验日志，并把元信息和每个 epoch 的指标放在一起。"""
    loaded = []
    for exp in EXPERIMENTS:
        records = parse_log(exp["log_path"])
        # 和 train.py 的 best checkpoint 保存逻辑保持一致：
        # 只要 val_acc 严格变高才更新 best；如果后面 val_acc 打平，不会覆盖旧 checkpoint。
        # 因此这里选择“第一次达到最高 val_acc 的 epoch”，而不是选择 val_loss 最低的 tie。
        best_val_acc = max(item["val_acc"] for item in records)
        best = next(item for item in records if item["val_acc"] == best_val_acc)
        loaded.append({**exp, "records": records, "best": best})
    return loaded


def setup_matplotlib():
    """延迟导入 matplotlib，避免用户只查看脚本时就要求必须安装绘图库。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.facecolor"] = "white"
    return plt


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_dropout_seed_best_acc(experiments: list[dict], out_dir: Path) -> None:
    """图 1：横轴 dropout，纵轴 best validation accuracy，不同 seed 画成不同折线。"""
    plt = setup_matplotlib()

    grouped = defaultdict(list)
    for exp in experiments:
        grouped[exp["seed"]].append(exp)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for seed, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["dropout"])
        xs = [item["dropout"] for item in items]
        ys = [item["best"]["val_acc"] for item in items]
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=7, label=f"seed={seed}")

        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    ax.set_title("Dropout Ablation: Best Validation Accuracy", fontsize=13)
    ax.set_xlabel("Dropout")
    ax.set_ylabel("Best Val Acc")
    ax.set_xticks(sorted({exp["dropout"] for exp in experiments}))
    ax.set_ylim(0.935, 0.960)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir / "dropout_seed_best_val_acc.png")
    plt.close(fig)


def plot_epoch_curves(experiments: list[dict], out_dir: Path, metric: str) -> None:
    """图 2/3：按 epoch 展示 val_acc 或 val_loss 曲线，把不同 seed/dropout 合并到同一张图。"""
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    markers = {42: "o", 123: "s"}
    linestyles = {42: "-", 123: "--"}

    for exp in sorted(experiments, key=lambda item: (item["dropout"], item["seed"])):
        epochs = [item["epoch"] for item in exp["records"]]
        values = [item[metric] for item in exp["records"]]
        ax.plot(
            epochs,
            values,
            marker=markers.get(exp["seed"], "o"),
            linestyle=linestyles.get(exp["seed"], "-"),
            linewidth=2.0,
            markersize=4.8,
            label=f"seed={exp['seed']}, dropout={exp['dropout']}",
        )

    ax.set_title(f"{metric} over Epochs", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{metric}_curves_by_dropout.png")
    plt.close(fig)


def plot_ablation_table(experiments: list[dict], out_dir: Path) -> None:
    """图 4：把 best checkpoint 结果渲染成图片表格，方便直接放 PPT。"""
    plt = setup_matplotlib()

    rows = []
    for exp in sorted(experiments, key=lambda item: (item["seed"], item["dropout"])):
        best = exp["best"]
        rows.append(
            [
                f"{exp['dropout']:.1f}",
                str(exp["seed"]),
                str(best["epoch"]),
                f"{best['val_loss']:.4f}",
                f"{best['val_acc']:.4f}",
            ]
        )

    columns = ["Dropout", "Seed", "Best Epoch", "Val Loss", "Val Acc"]

    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_facecolor("#E8EEF8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F8FA")

    ax.set_title("Dropout Ablation Summary", fontsize=13, pad=16)
    save_figure(fig, out_dir / "dropout_ablation_table.png")
    plt.close(fig)


def draw_text_panel(
    ax,
    title: str,
    body: str,
    title_size: int = 15,
    body_size: int = 11,
) -> None:
    """用 matplotlib 画简单文字卡片，比依赖 graphviz 更轻量。"""
    ax.axis("off")
    ax.text(0.02, 0.95, title, fontsize=title_size, weight="bold", va="top")
    ax.text(
        0.02,
        0.86,
        body,
        fontsize=body_size,
        va="top",
        family="monospace",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#FAFBFC",
            "edgecolor": "#B8C0CC",
        },
    )


def plot_project_structure(out_dir: Path) -> None:
    """图 5：项目目录结构图。"""
    plt = setup_matplotlib()
    body = """DART-UMMs
├── configs/       experiment YAMLs
├── data/          Dataset / DataLoader
├── models/        MLPRouter
├── utils/         config / seed / metrics
├── scripts/       reports and figures
├── results/       ablation notes
├── train.py       training entry
└── eval.py        evaluation entry"""

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    draw_text_panel(ax, "Current Project Structure", body)
    save_figure(fig, out_dir / "project_structure.png")
    plt.close(fig)


def plot_training_pipeline(out_dir: Path) -> None:
    """图 6：训练闭环流程图。"""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.axis("off")

    steps = [
        "YAML\nConfig",
        "Synthetic\nDataset",
        "MLP\nRouter",
        "Train\nLoop",
        "Validation",
        "Best / Last\nCheckpoint",
        "Eval +\nReports",
    ]

    x_positions = [i / (len(steps) - 1) for i in range(len(steps))]
    y = 0.55

    for idx, (x, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#EEF4FF",
                "edgecolor": "#5577AA",
            },
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.055, y),
                xytext=(x + 0.055, y),
                arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#333333"},
            )

    ax.set_title("Mini TableDART Router Training Pipeline", fontsize=14, weight="bold")
    save_figure(fig, out_dir / "training_pipeline.png")
    plt.close(fig)


def plot_official_mapping(out_dir: Path) -> None:
    """图 7：我们的 mini 项目与官方 TableDART 代码的对应关系。"""
    plt = setup_matplotlib()
    body = """Mini Project                    Official TableDART
------------------------------------------------------------
models/router.py             -> models/gating_network.py
data/dataloader.py           -> data/dataloader.py
train.py                     -> train.py
eval.py                      -> evaluate.py / inference.py
configs/*.yaml               -> project_config/config.py
next target                  -> models/dynamic_selector.py"""

    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    draw_text_panel(ax, "Mini Project vs Official TableDART", body)
    save_figure(fig, out_dir / "mini_to_official_mapping.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiments = load_experiments()

    plot_dropout_seed_best_acc(experiments, args.out_dir)
    plot_epoch_curves(experiments, args.out_dir, metric="val_acc")
    plot_epoch_curves(experiments, args.out_dir, metric="val_loss")
    plot_ablation_table(experiments, args.out_dir)
    plot_project_structure(args.out_dir)
    plot_training_pipeline(args.out_dir)
    plot_official_mapping(args.out_dir)

    print("\nAll meeting figures generated.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
