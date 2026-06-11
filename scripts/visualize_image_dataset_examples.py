"""
生成 MMTab / TableDART 图片数据集样例卡片图。

这个脚本是给组会 PPT 用的“数据集直观展示”脚本。

上一阶段我们已经用 scripts/inspect_image_inputs.py 验证了：

    test_data.jsonl 里的 image 字段
    -> data/images/MMTab-eval/all_test_image 里的真实表格图片
    -> 官方 image_prompt.py 构造出来的 VLM prompt

三者能够对齐，并且当前 448 条样本的图片都能找到。

但是终端输出不适合直接放 PPT，所以这个脚本会做一张更直观的图片：

    每个任务类别选择 1 条样本
    每条样本做成一张卡片：
        - Category
        - question_id
        - image 文件名
        - question 文本
        - 表格图片缩略图

默认覆盖 7 个 TableDART / MMTab 测试类别：
    WTQ_for_TQA
    TabFact_for_TFV
    FeTaQA_for_TQA
    TAT-QA_for_TQA
    HiTab_for_TQA
    TABMWP_for_TQA
    InfoTabs_for_TFV

运行方式：
    python scripts/visualize_image_dataset_examples.py

如果你的图片目录不在默认位置，可以传入：
    python scripts/visualize_image_dataset_examples.py --image-root path/to/all_test_image

输出文件：
    results/figures/image_dataset_examples.png

这张图在组会里可以说明：
    “当前已经从官方 jsonl 对齐到了真实表格图片，Image-only path 的输入数据已经准备好，
     下一步可以接 Ovis2 提取 6144 维 image feature。”
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from textwrap import wrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def resolve_project_path(path_str: str | Path) -> Path:
    """把配置里的相对路径解析成相对于项目根目录的绝对路径。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    加载一个尽量稳定的字体。

    为什么要单独写字体函数？
    PIL 默认字体比较小，放到 PPT 里容易看不清。
    Windows 上通常有 Arial / Arial Bold；如果找不到，就退回 PIL 默认字体。
    """
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ]
        )

    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def load_jsonl_samples(
    data_path: Path,
    max_samples: int | None,
    samples_per_category: int | None,
) -> list[dict[str, Any]]:
    """
    读取 jsonl，并复刻 RealFormatMockFeatureDataset 的采样逻辑。

    这样生成的样例和训练/检查脚本使用的是同一批 448 条样本：
        7 个类别 * 每类 64 条
    """
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    all_samples = []
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                all_samples.append(json.loads(line))

    if samples_per_category is not None:
        category_counts: dict[str, int] = {}
        sampled = []
        for item in all_samples:
            category = item.get("category", "")
            current_count = category_counts.get(category, 0)
            if current_count < samples_per_category:
                sampled.append(item)
                category_counts[category] = current_count + 1
    else:
        sampled = all_samples

    if max_samples is not None:
        sampled = sampled[:max_samples]

    return sampled


def select_examples_by_category(samples: list[dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
    """
    每个 category 选第一条样本。

    用 OrderedDict 是为了保留 jsonl 原本的类别出现顺序，
    这样和 inspect_image_inputs.py 的终端展示顺序一致。
    """
    examples: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in samples:
        category = item.get("category", "")
        if category not in examples:
            examples[category] = item
    return examples


def resolve_image_path(image_name: str, image_root: Path) -> Path:
    """
    把 jsonl 里的 image 文件名转成本地图片路径。

    当前 MMTab eval 图片包解压后的结构是：
        data/images/MMTab-eval/all_test_image/xxx.jpg

    jsonl 中保存的是：
        xxx.jpg

    所以最终路径就是：
        image_root / image_name
    """
    image_path = image_root / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")
    return image_path


def fit_image_to_box(image_path: Path, box_size: tuple[int, int]) -> Image.Image:
    """
    读取并缩放表格图片，让它完整显示在卡片缩略图区域中。

    使用 ImageOps.contain 而不是直接 resize：
        - contain 会保持原图宽高比；
        - 不会把宽表格或高表格强行拉伸变形；
        - 空白区域由后面的白色背景补齐。
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        thumbnail = ImageOps.contain(image, box_size, method=Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", box_size, "#FFFFFF")
    paste_x = (box_size[0] - thumbnail.width) // 2
    paste_y = (box_size[1] - thumbnail.height) // 2
    canvas.paste(thumbnail, (paste_x, paste_y))
    return canvas


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_chars_per_line: int,
    max_lines: int,
    line_spacing: int,
) -> int:
    """
    绘制自动换行文本，并返回绘制结束后的 y 坐标。

    PIL 不会像 HTML/CSS 那样自动换行，所以这里用 textwrap.wrap 按字符数粗略切行。
    对英文问题来说，这种方式足够用于 PPT 样例卡片。
    """
    x, y = position
    lines = wrap(text, width=max_chars_per_line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_spacing if hasattr(font, "size") else 18
    return y


def draw_card(
    item: dict[str, Any],
    image_root: Path,
    card_size: tuple[int, int],
) -> Image.Image:
    """
    绘制单张样例卡片。

    卡片结构：
        顶部色条：category
        文本区：question_id / image name / question
        图片区：表格图片缩略图
    """
    card_w, card_h = card_size
    card = Image.new("RGB", card_size, "#FBFAF6")
    draw = ImageDraw.Draw(card)

    font_title = load_font(24, bold=True)
    font_meta = load_font(17)
    font_question = load_font(18)

    category = item.get("category", "")
    question_id = item.get("question_id", "")
    question = item.get("question", "")
    image_name = item.get("image", "")
    image_path = resolve_image_path(image_name, image_root)

    # 卡片边框和顶部标题色条。
    draw.rounded_rectangle(
        (0, 0, card_w - 1, card_h - 1),
        radius=18,
        fill="#FBFAF6",
        outline="#B8B0A0",
        width=2,
    )
    draw.rounded_rectangle(
        (0, 0, card_w - 1, 58),
        radius=18,
        fill="#D9E7D2",
        outline="#D9E7D2",
    )
    # 顶部色条下边缘补一个矩形，避免圆角影响视觉分割。
    draw.rectangle((0, 36, card_w - 1, 58), fill="#D9E7D2")
    draw.text((22, 16), category, font=font_title, fill="#1E3D2F")

    y = 78
    draw.text((22, y), f"ID: {question_id}", font=font_meta, fill="#333333")
    y += 26
    draw.text((22, y), f"Image: {image_name}", font=font_meta, fill="#555555")
    y += 34
    y = draw_wrapped_text(
        draw=draw,
        text=f"Q: {question}",
        position=(22, y),
        font=font_question,
        fill="#1F1F1F",
        max_chars_per_line=48,
        max_lines=3,
        line_spacing=5,
    )

    # 图片区固定大小。不同表格宽高差异很大，所以统一放入白底框中。
    image_box_w = card_w - 44
    image_box_h = card_h - y - 32
    image_box_h = max(image_box_h, 150)
    thumb = fit_image_to_box(image_path, (image_box_w, image_box_h))

    image_x = 22
    image_y = card_h - image_box_h - 22
    draw.rounded_rectangle(
        (image_x - 1, image_y - 1, image_x + image_box_w + 1, image_y + image_box_h + 1),
        radius=10,
        fill="#FFFFFF",
        outline="#D0C8B8",
        width=1,
    )
    card.paste(thumb, (image_x, image_y))

    return card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 TableDART/MMTab 图片数据集样例卡片图。",
    )
    parser.add_argument(
        "--config",
        default="configs/router_real_format_cached_text_question_mock.yaml",
        help="用于读取 jsonl 路径和采样参数的配置文件。",
    )
    parser.add_argument(
        "--image-root",
        default="data/images/MMTab-eval/all_test_image",
        help="MMTab eval 解压后的真实图片目录。",
    )
    parser.add_argument(
        "--output",
        default="results/figures/image_dataset_examples.png",
        help="输出图片路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))

    data_path = resolve_project_path(config["data"]["data_path"])
    image_root = resolve_project_path(args.image_root)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl_samples(
        data_path=data_path,
        max_samples=config["data"].get("max_samples"),
        samples_per_category=config["data"].get("samples_per_category"),
    )
    examples = select_examples_by_category(samples)

    if not image_root.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_root}")

    # 7 个类别用 4 x 2 网格展示，最后一个格子留空。
    card_size = (540, 420)
    cols = 4
    rows = 2
    gap = 26
    margin_x = 34
    margin_y = 120
    title_h = 90

    canvas_w = margin_x * 2 + cols * card_size[0] + (cols - 1) * gap
    canvas_h = title_h + rows * card_size[1] + (rows - 1) * gap + 42
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#F4F1E9")
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(42, bold=True)
    font_subtitle = load_font(22)

    draw.text(
        (margin_x, 30),
        "MMTab Evaluation Table Images: Matched Examples",
        font=font_title,
        fill="#1E3D2F",
    )
    draw.text(
        (margin_x, 78),
        "Each card links one real-format jsonl sample to its table image. Image path check: 448 / 448 matched.",
        font=font_subtitle,
        fill="#5A5A5A",
    )

    for index, item in enumerate(examples.values()):
        row = index // cols
        col = index % cols
        x = margin_x + col * (card_size[0] + gap)
        y = margin_y + row * (card_size[1] + gap)

        card = draw_card(item=item, image_root=image_root, card_size=card_size)
        canvas.paste(card, (x, y))

    canvas.save(output_path)

    print(f"已生成图片数据集样例图: {output_path}")
    print(f"样例类别数: {len(examples)}")
    print(f"图片根目录: {image_root}")


if __name__ == "__main__":
    main()
