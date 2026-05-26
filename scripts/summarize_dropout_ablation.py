import argparse
import re
from pathlib import Path


# 训练日志中的每一行长这样:
# Epoch 02 | train_loss=0.0072 train_acc=1.0000 | val_loss=0.1363 val_acc=0.9458
#
# 这里用正则表达式把 epoch、train_loss、train_acc、val_loss、val_acc 抓出来。
# 括号 (...) 表示“我要提取这一段内容”，后面会通过 match.group(...) 读取。
LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


# 每一项表示一组实验:
# (dropout 值, seed 值, config 文件路径, log 文件路径)
#
# 这个列表就是当前 dropout ablation 的“实验清单”。
# 如果后面新增 seed=777 或 dropout=0.5，只需要在这里继续加一行。
DEFAULT_EXPERIMENTS = [
    ("0.0", "42", Path("configs/router_dropout_0.yaml"), Path("logs/router_dropout_0_log.txt")),
    ("0.1", "42", Path("configs/router_sanity.yaml"), Path("logs/router_sanity_log.txt")),
    ("0.3", "42", Path("configs/router_dropout_03.yaml"), Path("logs/router_dropout_03_log.txt")),
    (
        "0.0",
        "123",
        Path("configs/router_dropout_0_seed123.yaml"),
        Path("logs/router_dropout_0_seed123_log.txt"),
    ),
    (
        "0.3",
        "123",
        Path("configs/router_dropout_03_seed123.yaml"),
        Path("logs/router_dropout_03_seed123_log.txt"),
    ),
]


def parse_log(log_path: Path) -> list[dict]:
    """读取一个训练日志文件，并解析出每个 epoch 的指标。"""
    records = []

    # read_text 会把整个 txt 文件读成一个字符串。
    # splitlines 会按行切开，得到一个 list[str]。
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.search(line)

        # 如果某一行不是 epoch 记录，就跳过。
        if match is None:
            continue

        # match.group(1) 对应正则里的第 1 个括号，也就是 epoch。
        # 后面的 group(2) ~ group(5) 分别是 train_loss/train_acc/val_loss/val_acc。
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
    """从所有 epoch 中选出 best checkpoint 对应的记录。"""
    # 选择规则:
    # 1. 优先选 val_acc 最高的 epoch
    # 2. 如果 val_acc 一样，则选 val_loss 更低的 epoch
    #
    # key=lambda item: (...) 表示告诉 max 函数“按什么标准比较”。
    # -item["val_loss"] 的意思是 val_loss 越小越好。
    return max(records, key=lambda item: (item["val_acc"], -item["val_loss"]))


def build_markdown(rows: list[dict]) -> str:
    """把实验结果拼成 Markdown 文本。"""
    # Markdown 表格本质上就是普通字符串。
    # 比如下面两行:
    # | A | B |
    # |---|---|
    # 在 Markdown 里就会被渲染成表格。
    #
    # 这里先用一个 list 保存每一行文本，最后用 "\n".join(lines)
    # 把它们拼成一个完整 Markdown 文件。
    lines = [
        "# Dropout Ablation Summary",
        "",
        "Table: validation performance under different dropout and seed settings.",
        "",
        "| Dropout | Seed | Config | Best Epoch | Train Acc | Val Loss | Val Acc |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        # f-string 用来把变量插入字符串。
        # :.4f 表示保留 4 位小数。
        lines.append(
            f"| {row['dropout']} | {row['seed']} | `{row['config']}` | {row['epoch']} | "
            f"{row['train_acc']:.4f} | {row['val_loss']:.4f} | {row['val_acc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Observation:",
            "",
            "- Under seed 42, dropout=0.3 reaches the highest validation accuracy.",
            "- Under seed 123, dropout=0.0 performs better than dropout=0.3 in validation accuracy.",
            "- The ranking changes across seeds, so the current synthetic setting does not support a strong claim that one dropout value is consistently better.",
            "",
            "Note: This table is generated from synthetic route-label experiments. It should be used only as a sanity-check ablation record, not as a real TableDART benchmark result.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Summarize dropout ablation logs into a Markdown report.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "dropout_ablation_summary.md",
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for dropout, seed, config_path, log_path in DEFAULT_EXPERIMENTS:
        records = parse_log(log_path)
        best = select_best_record(records)

        # **best 会把 best 这个字典里的键值展开，合并进新的 row 字典。
        rows.append(
            {
                "dropout": dropout,
                "seed": seed,
                "config": str(config_path),
                **best,
            }
        )

    # 确保 results/ 目录存在。
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 把 Markdown 文本写入文件。
    args.out.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Saved dropout ablation summary to: {args.out}")


if __name__ == "__main__":
    main()
