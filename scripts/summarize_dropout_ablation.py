import argparse
import re
from pathlib import Path


LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)

DEFAULT_EXPERIMENTS = [
    ("0.0", Path("configs/router_dropout_0.yaml"), Path("logs/router_dropout_0_log.txt")),
    ("0.1", Path("configs/router_sanity.yaml"), Path("logs/router_sanity_log.txt")),
    ("0.3", Path("configs/router_dropout_03.yaml"), Path("logs/router_dropout_03_log.txt")),
]


def parse_log(log_path: Path) -> list[dict]:
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
        raise ValueError(f"No valid epoch records found in {log_path}")
    return records


def select_best_record(records: list[dict]) -> dict:
    return max(records, key=lambda item: (item["val_acc"], -item["val_loss"]))


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# Dropout Ablation Summary",
        "",
        "Table: validation performance under different dropout settings.",
        "",
        "| Dropout | Config | Best Epoch | Train Acc | Val Loss | Val Acc |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['dropout']} | `{row['config']}` | {row['epoch']} | "
            f"{row['train_acc']:.4f} | {row['val_loss']:.4f} | {row['val_acc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Note: This table is generated from synthetic route-label experiments. It should be used only as a sanity-check ablation record, not as a real TableDART benchmark result.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dropout ablation logs into a Markdown table.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "dropout_ablation_table.md",
        help="Output Markdown table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for dropout, config_path, log_path in DEFAULT_EXPERIMENTS:
        records = parse_log(log_path)
        best = select_best_record(records)
        rows.append(
            {
                "dropout": dropout,
                "config": str(config_path),
                **best,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Saved dropout ablation table to: {args.out}")


if __name__ == "__main__":
    main()
