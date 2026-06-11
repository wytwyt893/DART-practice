"""
绘制“TableDART 三路特征接入进度”的 Train/Val 准确率折线图。

这个脚本是给组会 PPT 用的可视化脚本，不是训练脚本。

当前复现主线里，TableDART 的 gating network 输入由三类特征拼接而成：

    text_feature     [B, 3584]  -> Table-as-Text Expert / TableGPT2
    image_feature    [B, 6144]  -> Table-as-Image Expert / Ovis2
    question_feature [B, 384]   -> Question Encoder / all-MiniLM-L6-v2

我们现在不是一次性把三路都替换成真实特征，而是按阶段逐步替换：

    Stage A: 全部 mock feature
    Stage B: question_feature 换成真实 MiniLM embedding
    Stage C: text_feature 换成真实 TableGPT2 feature，question_feature 保持真实
    Stage D: image_feature 后续再换成真实 Ovis2 feature

因此这张图想表达的不是“最终性能排名”，而是：

    我们已经把 Question 路径和 Text 路径从 mock 推进到了真实专家特征；
    Image 路径目前仍然是 mock，正在准备表格图片和 Ovis2 特征提取。

为什么不直接画一个总表？
组会上折线图更直观：
    - 横轴是 epoch；
    - 纵轴是 accuracy；
    - 每个子图对应一条 feature path；
    - 颜色表示 mock / real feature；
    - 线型表示 train / validation split。

这样一张图能同时看两个问题：
    1. 训练集是否能被模型学会；
    2. 验证集是否同步提升，是否出现明显 train-val gap。

注意一个非常重要的解释点：
当前 route_label 仍然是基于 category 的临时 mock/heuristic 标签，
不是论文里根据 Text Expert / Image Expert / Fusion 答案正确性构造出来的真实 policy label。
所以真实 feature 接入后 val acc 下降并不等于“真实 feature 不好”，
而是说明真实特征分布和人工 mock 标签之间出现了不匹配。

运行方式：
    python scripts/plot_feature_path_val_acc_curves.py

输出文件：
    results/figures/feature_path_train_val_acc_curves.png
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 训练日志的标准格式来自 train_real_format_mock.py：
# Epoch 01 | train_loss=0.7986 train_acc=0.7430 | val_loss=0.3792 val_acc=0.9667
#
# 这里用正则表达式从每一行中提取：
# epoch / train_loss / train_acc / val_loss / val_acc
LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+"
    r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+\|\s+"
    r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)"
)


def parse_log(log_path: Path) -> dict[str, list[float]]:
    """
    从训练日志中解析每个 epoch 的指标。

    返回值是一个字典，例如：
        {
            "epoch": [1, 2, 3, ...],
            "train_loss": [...],
            "train_acc": [...],
            "val_loss": [...],
            "val_acc": [...],
        }

    本图主要使用 val_acc，但保留其它字段是为了后续扩展 loss 图时不用重写解析逻辑。
    """
    if not log_path.exists():
        raise FileNotFoundError(f"找不到训练日志: {log_path}")

    metrics: dict[str, list[float]] = {
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
        raise ValueError(f"日志中没有解析到有效 epoch 指标: {log_path}")

    return metrics


def best_point(metrics: dict[str, list[float]]) -> tuple[int, float]:
    """
    找到验证集准确率最高的点。

    返回：
        best_epoch, best_val_acc

    图中会用星号标出这个点，方便组会时快速说明“最好结果在哪个 epoch”。
    """
    best_index = max(range(len(metrics["val_acc"])), key=lambda index: metrics["val_acc"][index])
    return int(metrics["epoch"][best_index]), float(metrics["val_acc"][best_index])


# 每个 panel 表示一条 feature path。
#
# 这里的对比设计要特别小心：
#
# 1. Question path:
#    Paper-dim mock vs Real question embedding
#    这一步主要观察 question_feature 从 mock signal 换成 MiniLM 真实语义向量后的曲线变化。
#
# 2. Text path:
#    Text mock baseline 使用 “Real question embedding” 实验。
#    Real text 使用 “Real text + real question” 实验。
#    这样做是为了尽量固定 question_feature 已经是真实的，只观察 text_feature 被替换后的变化。
#
# 3. Image path:
#    目前还没有 Ovis2 image_features.pt，所以只能画当前 image mock 的曲线。
#    图中会额外标注 “Ovis2 pending”，提醒这条路径还没完成真实特征接入。
PANELS = [
    {
        "title": "Question Feature Path",
        "ylabel": "Accuracy",
        "series": [
            {
                "label": "Mock",
                "log": PROJECT_ROOT / "logs" / "router_real_format_paper_dims_mock_log.txt",
                "color": "#2E6F9E",
                "marker": "o",
            },
            {
                "label": "Real MiniLM",
                "log": PROJECT_ROOT / "logs" / "router_real_format_real_question_mock_log.txt",
                "color": "#C75146",
                "marker": "o",
            },
        ],
        "note": "Q: mock -> MiniLM [384]",
    },
    {
        "title": "Text Feature Path",
        "ylabel": "Accuracy",
        "series": [
            {
                "label": "Mock text",
                "log": PROJECT_ROOT / "logs" / "router_real_format_real_question_mock_log.txt",
                "color": "#7A8A99",
                "marker": "s",
            },
            {
                "label": "Real TableGPT2",
                "log": PROJECT_ROOT / "logs" / "router_real_format_cached_text_question_mock_log.txt",
                "color": "#D8942F",
                "marker": "s",
            },
        ],
        "note": "T: mock -> TableGPT2 [3584]",
    },
    {
        "title": "Image Feature Path",
        "ylabel": "Accuracy",
        "series": [
            {
                "label": "Mock image",
                "log": PROJECT_ROOT / "logs" / "router_real_format_cached_text_question_mock_log.txt",
                "color": "#4D8B57",
                "marker": "^",
            },
        ],
        "note": "I: Ovis2 [6144] pending",
    },
]


def plot_panel(ax: plt.Axes, panel: dict[str, object]) -> None:
    """
    绘制单个子图。

    同一条实验曲线会画两次：
        - train_acc: 实线
        - val_acc: 虚线

    这样你在组会上能解释：
        “模型在训练集上很快学到 mock/heuristic 标签，
         但验证集表现取决于真实 feature 与当前 route label 是否匹配。”
    """
    for series in panel["series"]:  # type: ignore[index]
        metrics = parse_log(series["log"])  # type: ignore[index]

        for split_name, metric_key, linestyle, alpha in [
            ("Train", "train_acc", "-", 0.86),
            ("Val", "val_acc", "--", 1.0),
        ]:
            ax.plot(
                metrics["epoch"],
                metrics[metric_key],
                marker=series["marker"],  # type: ignore[index]
                linestyle=linestyle,
                linewidth=2.35,
                markersize=4.8,
                color=series["color"],  # type: ignore[index]
                alpha=alpha,
                label=f"{series['label']} {split_name}",  # type: ignore[index]
            )

        # 用星号标记该实验曲线的 best validation accuracy。
        # 只标 validation，不标 train，是因为 train 很快到 1.0；
        # 真正汇报时更关心验证集上的最好点。
        best_epoch, best_acc = best_point(metrics)
        ax.scatter(
            [best_epoch],
            [best_acc],
            marker="*",
            s=140,
            color=series["color"],  # type: ignore[index]
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{best_acc:.3f}",
            xy=(best_epoch, best_acc),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=series["color"],  # type: ignore[index]
        )

    ax.set_title(panel["title"], fontsize=13, fontweight="bold")  # type: ignore[index]
    ax.set_xlabel("Epoch")
    ax.set_ylabel(panel["ylabel"])  # type: ignore[arg-type]
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.50, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.8, loc="lower right", frameon=True)

    # 在子图左下角加一行状态说明。
    # 这能帮助听众理解每个 panel 到底替换了哪一路特征。
    ax.text(
        0.03,
        0.04,
        panel["note"],  # type: ignore[index]
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#F6F1E8",
            "edgecolor": "#D8CFBF",
            "alpha": 0.95,
        },
    )


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_path_train_val_acc_curves.png"

    # DejaVu Sans 是 matplotlib 默认随包字体，英文和数字显示稳定。
    # 这里图中文字主要用英文，是为了避免不同电脑上中文字体缺失导致乱码。
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)

    for ax, panel in zip(axes, PANELS):
        plot_panel(ax, panel)

    fig.suptitle(
        "Train/Validation Accuracy Curves by Feature Path",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        (
            "Note: current route labels are still heuristic/mock. "
            "Solid lines are train accuracy; dashed lines are validation accuracy. "
            "A larger train-val gap after real feature injection indicates feature-label mismatch, not final TableDART performance."
        ),
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - 让组会脚本失败时给出明确提示
        print(f"生成图像失败: {error}", file=sys.stderr)
        raise
