import argparse
from pathlib import Path


# 这个脚本用于画当前 mini 实验中的 MLPRouter 网络结构图。
# 目标不是精确画出每一个神经元，而是给组会 PPT 一个直观的结构示意：
# 10112 维输入特征 -> Linear -> ReLU -> Dropout -> Linear -> 3 路路由输出。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the current MLP router architecture.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "figures" / "mlp_router_architecture.png",
        help="Output image path.",
    )
    parser.add_argument("--input-dim", type=int, default=10112)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-routes", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def setup_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError as exc:
        raise RuntimeError("matplotlib is required. Install it with: pip install matplotlib") from exc

    plt.rcParams["figure.facecolor"] = "white"
    return plt, FancyBboxPatch, FancyArrowPatch


def add_box(ax, FancyBboxPatch, x, y, width, height, title, subtitle, facecolor):
    """画一个带圆角的模块框。"""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=1.8,
        edgecolor="#263238",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x + width / 2, y + height * 0.34, subtitle, ha="center", va="center", fontsize=9.5)


def add_arrow(ax, FancyArrowPatch, start, end, label=None):
    """画模块之间的数据流箭头。"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=2.0,
        color="#333333",
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.06, label, ha="center", va="bottom", fontsize=8.5, color="#444444")


def add_route_outputs(ax, x, y):
    """画三个 route 输出，表示 Text / Image / Fusion 三路选择。"""
    routes = [
        ("Route 0", "Text-only", "#E8F5E9"),
        ("Route 1", "Image-only", "#E3F2FD"),
        ("Route 2", "Fusion", "#FFF3E0"),
    ]
    for idx, (title, subtitle, color) in enumerate(routes):
        yy = y - idx * 0.18
        ax.text(
            x,
            yy,
            f"{title}: {subtitle}",
            ha="left",
            va="center",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": color,
                "edgecolor": "#546E7A",
                "linewidth": 1.2,
            },
        )


def plot_architecture(args: argparse.Namespace) -> None:
    plt, FancyBboxPatch, FancyArrowPatch = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(14.2, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Current Mini TableDART Router: Two-layer MLP Gating Network",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.5,
        0.895,
        "Synthetic gate feature is mapped to route logits, then used to select Text / Image / Fusion path.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#455A64",
    )

    y = 0.54
    height = 0.19
    width = 0.115

    boxes = [
        (0.045, "Input Feature", f"[B, {args.input_dim}]", "#F5F5F5"),
        (0.205, "Linear", f"{args.input_dim} -> {args.hidden_dim}", "#E3F2FD"),
        (0.365, "ReLU", "non-linearity", "#E8F5E9"),
        (0.515, "Dropout", f"p = {args.dropout}", "#FFF8E1"),
        (0.675, "Linear", f"{args.hidden_dim} -> {args.num_routes}", "#E3F2FD"),
    ]

    for x, title, subtitle, color in boxes:
        add_box(ax, FancyBboxPatch, x, y, width, height, title, subtitle, color)

    for idx in range(len(boxes) - 1):
        start_x = boxes[idx][0] + width
        end_x = boxes[idx + 1][0]
        add_arrow(ax, FancyArrowPatch, (start_x + 0.012, y + height / 2), (end_x - 0.012, y + height / 2))

    add_arrow(
        ax,
        FancyArrowPatch,
        (0.675 + width + 0.01, y + height / 2),
        (0.865, y + height / 2),
        label=None,
    )
    ax.text(0.82, y + height / 2 + 0.085, "logits -> softmax", ha="center", va="bottom", fontsize=8.5, color="#444444")
    add_route_outputs(ax, 0.86, 0.70)

    note_style = {
        "boxstyle": "round,pad=0.55",
        "facecolor": "#FAFBFC",
        "edgecolor": "#B8C0CC",
        "linewidth": 1.2,
    }
    ax.text(
        0.08,
        0.22,
        "Current scope\n"
        "Sanity-check reproduction of TableDART's\n"
        "lightweight gating network, not the full expert system.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#37474F",
        bbox=note_style,
        linespacing=1.45,
    )

    ax.text(
        0.55,
        0.22,
        "Next upgrade\n"
        "Replace synthetic feature with concatenated\n"
        "text / image / question gate features.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#37474F",
        bbox=note_style,
        linespacing=1.45,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MLP router architecture figure to: {args.out}")


def main() -> None:
    args = parse_args()
    plot_architecture(args)


if __name__ == "__main__":
    main()
