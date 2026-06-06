from pathlib import Path
import re

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXPERIMENTS = {
    "Paper-dim Mock": {
        "log_path": PROJECT_ROOT / "logs" / "router_real_format_paper_dims_mock_log.txt",
        "text_feature": "Mock",
        "image_feature": "Mock",
        "question_feature": "Mock + signal",
        "note": "Paper dims, all features synthetic",
    },
    "Real Question": {
        "log_path": PROJECT_ROOT / "logs" / "router_real_format_real_question_mock_log.txt",
        "text_feature": "Mock",
        "image_feature": "Mock",
        "question_feature": "MiniLM real embedding",
        "note": "Question feature replaced by real encoder output",
    },
}


LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+"
    r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+"
    r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


def parse_log(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)
        if match is None:
            continue
        epoch, train_loss, train_acc, val_loss, val_acc = match.groups()
        rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

    if not rows:
        raise ValueError(f"No valid training rows found in: {log_path}")

    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda row: row["val_acc"])


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage5_experiment_comparison_table.png"

    table_rows = []
    for name, info in EXPERIMENTS.items():
        rows = parse_log(info["log_path"])
        best = best_row(rows)
        table_rows.append(
            [
                name,
                info["text_feature"],
                info["image_feature"],
                info["question_feature"],
                "3584 + 6144 + 384",
                f"{int(best['epoch'])}",
                f"{best['val_acc']:.4f}",
                f"{best['val_loss']:.4f}",
                info["note"],
            ]
        )

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(18, 5.4))
    ax.axis("off")

    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Experiment",
            "Text Feature",
            "Image Feature",
            "Question Feature",
            "Gate Input Dim",
            "Best Epoch",
            "Best Val Acc",
            "Best Val Loss",
            "Meaning",
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.13, 0.10, 0.10, 0.18, 0.15, 0.09, 0.10, 0.11, 0.24],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9B9488")
        cell.set_linewidth(1.1)
        if row == 0:
            cell.set_facecolor("#D9E8D4")
            cell.set_text_props(weight="bold", color="#1F3326")
        else:
            cell.set_facecolor("#FBFAF6" if row % 2 == 1 else "#F2EFE7")
            if col == 0:
                cell.set_text_props(weight="bold")
            if col == 6:
                cell.set_text_props(weight="bold", color="#0B6E4F")

    fig.suptitle(
        "Stage 5 Experiment Comparison: Mock Feature vs Real Question Embedding",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.08,
        "Both experiments use official-format data and paper-aligned gate dimensions. "
        "Real Question removes the artificial signal from question_feature and uses MiniLM [384] embeddings.",
        ha="center",
        fontsize=11,
        color="#4D4A45",
    )

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    print(f"Saved table figure to: {output_path}")


if __name__ == "__main__":
    main()
