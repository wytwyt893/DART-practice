"""
检查 TableDART 视觉路径的“输入端”是否准备正确。

本脚本对应当前复现计划里的 Stage 5.4A：
在真正加载 Ovis2 / LLaVA 这类视觉语言模型之前，先检查官方数据里和视觉路径有关的
几个关键字段能不能串起来。

为什么需要这个脚本？
之前我们已经完成了：
    1. question -> MiniLM -> question_features.pt
    2. table + question -> Text Expert prompt -> TableGPT2 -> text_features.pt

接下来要进入 image-only / VLM 路径。视觉路径比文本路径更容易踩坑，因为它除了 prompt
以外，还依赖真实图片文件：
    jsonl 里的 image 字段
    -> 本地图片路径
    -> PIL/processor 能否打开图片
    -> prompt_for_vlm_expert
    -> Ovis2 / LLaVA visual tokenizer
    -> image_features.pt

本脚本只做“输入检查”，不会：
    - 加载 Ovis2
    - 加载 LLaVA
    - 训练模型
    - 生成 image_features.pt

它会检查：
    1. 当前 config 指向哪个 jsonl 数据文件；
    2. 数据集中 image 字段是否存在；
    3. 按常见目录候选能否找到对应图片文件；
    4. 如果图片存在，是否能用 Pillow 打开并读取尺寸；
    5. 按官方 TableDART 的 image_prompt.py 构造 VLM prompt。

运行方式：
    python scripts/inspect_image_inputs.py

如果你已经知道真实图片目录在哪里，可以显式指定：
    python scripts/inspect_image_inputs.py --image-root path/to/table_images/test

读懂这个脚本以后，你应该能回答：
    - 官方 jsonl 里保存的是图片文件名还是完整路径？
    - 当前项目目录里是否真的有这些图片？
    - VLM path 的 prompt 和 Text Expert path 的 prompt 有什么区别？
    - 下一步接 Ovis2 之前缺的是代码、模型，还是数据文件？
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any


# 当我们用 `python scripts/inspect_image_inputs.py` 从项目根目录运行脚本时，
# Python 默认只会把 scripts/ 目录加入 import 搜索路径。
# 但 load_config 在 utils/ 目录下，所以这里手动把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from utils.config import load_config


# Windows PowerShell / VSCode 终端有时会使用非 UTF-8 编码。
# 这里尽量把 Python 标准输出切成 UTF-8，避免中文提示显示成乱码。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def print_section(title: str) -> None:
    """打印清晰的分隔标题，方便终端截图和复盘。"""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def shorten_text(text: str, max_chars: int) -> str:
    """
    截断过长 prompt。

    VLM prompt 虽然通常比 Text Expert 的 Markdown table prompt 短，但不同任务模板里
    仍然可能包含较长的说明。检查脚本只需要看结构是否正确，所以默认只展示前若干字符。
    """
    text = text.replace("\xa0", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[Prompt 过长，已截断展示]"


def resolve_project_path(path_str: str | Path) -> Path:
    """
    把配置里的路径统一解析成绝对路径。

    规则：
    - 如果传入的是绝对路径，直接使用；
    - 如果传入的是相对路径，就认为它是相对于项目根目录的路径。

    这样可以兼容 Windows 本地和 AutoDL/Linux 服务器两种运行环境。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl_samples(
    data_path: Path,
    max_samples: int | None,
    samples_per_category: int | None,
) -> list[dict[str, Any]]:
    """
    读取 jsonl，并复刻当前 RealFormatMockFeatureDataset 的采样规则。

    为什么不直接实例化 RealFormatMockFeatureDataset？
    因为这个脚本只检查图片输入，不应该依赖 text_features.pt 或 question_features.pt。
    如果为了看图片路径还要先准备特征缓存，脚本职责就混在一起了。

    当前采样规则：
    1. 先读取完整 jsonl；
    2. 如果设置 samples_per_category，就每个 category 最多取 N 条；
    3. 如果设置 max_samples，再从结果里截取前 max_samples 条。
    """
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    all_samples: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                all_samples.append(json.loads(line))

    if samples_per_category is not None:
        category_counts: dict[str, int] = {}
        sampled: list[dict[str, Any]] = []

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

    if not sampled:
        raise ValueError(f"没有从数据文件中读取到样本: {data_path}")

    return sampled


def load_official_image_prompt_module() -> ModuleType | None:
    """
    动态读取官方仓库里的 image_prompt.py。

    这样做的好处是：
    - 不需要我们手动复制一大段官方 prompt 模板；
    - 能保证 inspect 脚本看到的 VLM prompt 和官方仓库一致；
    - 如果以后官方模板有变化，我们只要更新官方仓库即可。

    如果本地没有 TableDART-Official，本函数返回 None，后面会使用一个兜底模板。
    """
    prompt_file = (
        PROJECT_ROOT
        / "TableDART-Official"
        / "TableDART"
        / "models"
        / "prompt"
        / "image_prompt.py"
    )

    if not prompt_file.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "official_tabledart_image_prompt",
        prompt_file,
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_vlm_prompt(
    item: dict[str, Any],
    prompt_module: ModuleType | None,
) -> tuple[str, str]:
    """
    根据样本 category 构造视觉专家 prompt。

    官方 TableDART 的逻辑大致是：
        category -> image_template_mapping[category] -> template.format(question=question)

    注意：
    这里构造的是“给视觉语言模型看的文字指令”，不是图片本身。
    视觉模型最终接收的是：
        image_path + prompt_for_vlm_expert
    """
    category = item.get("category", "")
    question = item.get("question", "")

    if prompt_module is not None:
        template_mapping = getattr(prompt_module, "image_template_mapping", {})
        default_template = getattr(
            prompt_module,
            "DEFAULT_IMAGE_PROMPT_TEMPLATE",
            "\nAnalyze the table image and answer the question.\n\nQuestion: {question}\n",
        )
        template = template_mapping.get(category, default_template)
        template_source = "官方 image_prompt.py"
    else:
        template = "\nAnalyze the table image and answer the question.\n\nQuestion: {question}\n"
        template_source = "兜底简化模板"

    return template.format(question=question), template_source


def collect_image_root_candidates(config: dict[str, Any], cli_image_root: str | None) -> list[Path]:
    """
    收集可能的图片根目录。

    官方 jsonl 的 image 字段通常只是文件名，例如：
        WTQ_203-csv_733.jpg

    因此我们需要一个 image_root，把它拼成：
        image_root / WTQ_203-csv_733.jpg

    当前 config 里还不一定有 image_root，所以这里同时检查：
    - 命令行传入的 --image-root；
    - config.data 里可能存在的 image_root / image_dir / table_image_dir；
    - 官方仓库常见的几个候选目录。
    """
    candidates: list[Path] = []

    if cli_image_root:
        candidates.append(resolve_project_path(cli_image_root))

    data_config = config.get("data", {})
    for key in ("image_root", "image_dir", "table_image_dir"):
        if data_config.get(key):
            candidates.append(resolve_project_path(data_config[key]))

    candidates.extend(
        [
            PROJECT_ROOT / "TableDART-Official" / "TableDART" / "data" / "test" / "table_images",
            PROJECT_ROOT / "TableDART-Official" / "TableDART" / "data" / "test" / "images",
            PROJECT_ROOT / "TableDART-Official" / "TableDART" / "data" / "table_images" / "test",
            PROJECT_ROOT / "TableDART-Official" / "TableDART" / "data" / "images" / "test",
            PROJECT_ROOT / "TableDART-Official" / "TableDART" / "data" / "test",
        ]
    )

    # 去重，同时保留顺序。这样命令行传入的路径优先级最高。
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    return unique_candidates


def resolve_image_path(image_name: str, image_roots: list[Path]) -> Path | None:
    """
    根据 image 字段和候选目录寻找真实图片文件。

    兼容两种情况：
    1. image 字段本身就是相对/绝对路径；
    2. image 字段只是文件名，需要和 image_root 拼接。
    """
    if not image_name:
        return None

    raw_path = Path(image_name)

    # 情况 1：jsonl 里的 image 字段本身已经是一个绝对路径。
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    # 情况 2：jsonl 里的 image 字段是相对项目根目录的路径。
    project_relative_path = PROJECT_ROOT / raw_path
    if project_relative_path.exists():
        return project_relative_path

    # 情况 3：jsonl 里的 image 字段只是文件名，需要到候选图片根目录里找。
    for root in image_roots:
        candidate = root / image_name
        if candidate.exists():
            return candidate

    return None


def get_image_info(image_path: Path) -> str:
    """
    尝试读取图片尺寸和颜色模式。

    如果 Pillow 没安装，或者图片文件本身损坏，都会返回可读的错误说明。
    这里不让脚本崩掉，是因为本阶段的目标是“定位问题”，不是强行继续训练。
    """
    try:
        from PIL import Image
    except ImportError:
        return "未安装 Pillow，暂时只能检查文件是否存在，不能读取图片尺寸"

    try:
        with Image.open(image_path) as image:
            return f"尺寸={image.size}, 模式={image.mode}"
    except Exception as error:  # noqa: BLE001 - 检查脚本需要把具体错误打印出来
        return f"图片存在，但 Pillow 打开失败: {error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 TableDART 视觉路径的图片字段、图片文件和 VLM prompt。",
    )
    parser.add_argument(
        "--config",
        default="configs/router_real_format_cached_text_question_mock.yaml",
        help="实验配置文件路径，默认使用当前 cached text + cached question 主线配置。",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="真实图片根目录。如果不传，脚本会尝试若干官方仓库常见目录。",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=1,
        help="每个任务类别展示多少条样例，默认每类 1 条。",
    )
    parser.add_argument(
        "--prompt-preview-chars",
        type=int,
        default=700,
        help="每条 VLM prompt 最多展示多少个字符。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)

    data_path = resolve_project_path(config["data"]["data_path"])
    max_samples = config["data"].get("max_samples")
    samples_per_category = config["data"].get("samples_per_category")

    print_section("1. 配置信息")
    print(f"配置文件: {config_path}")
    print(f"实验名称: {config['experiment_name']}")
    print(f"数据文件: {data_path}")
    print(f"max_samples: {max_samples}")
    print(f"samples_per_category: {samples_per_category}")

    prompt_module = load_official_image_prompt_module()
    if prompt_module is None:
        print("视觉 prompt 模板: 未找到官方 image_prompt.py，将使用兜底简化模板")
    else:
        print("视觉 prompt 模板: 已读取官方 TableDART image_prompt.py")

    image_roots = collect_image_root_candidates(config, args.image_root)

    print_section("2. 图片根目录候选")
    for index, root in enumerate(image_roots, start=1):
        status = "存在" if root.exists() else "不存在"
        print(f"{index}. [{status}] {root}")

    samples = load_jsonl_samples(
        data_path=data_path,
        max_samples=max_samples,
        samples_per_category=samples_per_category,
    )

    category_counter = Counter(item.get("category", "") for item in samples)
    image_names = [item.get("image", "") for item in samples]
    non_empty_image_count = sum(1 for image_name in image_names if image_name)
    unique_image_count = len(set(image_name for image_name in image_names if image_name))

    resolved_paths = [
        resolve_image_path(image_name, image_roots)
        for image_name in image_names
    ]
    found_image_count = sum(1 for path in resolved_paths if path is not None)
    missing_image_count = len(samples) - found_image_count

    print_section("3. 数据集图片字段概览")
    print(f"读取样本数: {len(samples)}")
    print(f"image 字段非空样本数: {non_empty_image_count}")
    print(f"唯一 image 文件名数量: {unique_image_count}")
    print(f"当前本地能找到的图片数: {found_image_count}")
    print(f"当前本地缺失的图片数: {missing_image_count}")

    print("\n任务类别分布:")
    for category, count in category_counter.items():
        print(f"  - {category}: {count}")

    print_section("4. 每类视觉输入样例")

    # OrderedDict 用来保留数据原始出现顺序，方便和前面文本 prompt 检查脚本对照。
    examples_by_category: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in samples:
        category = item.get("category", "")
        examples_by_category.setdefault(category, [])
        if len(examples_by_category[category]) < args.examples_per_category:
            examples_by_category[category].append(item)

    for category, examples in examples_by_category.items():
        for item in examples:
            question_id = item.get("question_id", "")
            question = item.get("question", "")
            image_name = item.get("image", "")
            image_path = resolve_image_path(image_name, image_roots)
            prompt, template_source = build_vlm_prompt(item, prompt_module)

            print("\n" + "-" * 88)
            print(f"问题 ID: {question_id}")
            print(f"任务类别: {category}")
            print(f"原始问题: {question}")
            print(f"jsonl image 字段: {image_name}")

            if image_path is None:
                print("图片解析结果: 未找到本地图片文件")
            else:
                print(f"图片解析结果: {image_path}")
                print(f"图片读取检查: {get_image_info(image_path)}")

            print(f"VLM prompt 模板来源: {template_source}")
            print(f"VLM prompt 字符数: {len(prompt)}")
            print("-" * 88)
            print(shorten_text(prompt, args.prompt_preview_chars))

    print_section("5. 检查结论")
    if found_image_count == len(samples):
        print("通过: 当前样本的图片文件都能在本地找到。")
        print("通过: 已能按官方 image_prompt.py 构造 prompt_for_vlm_expert。")
        print("下一步: 可以准备 Ovis2/LLaVA 的离线 image feature 提取脚本。")
    elif found_image_count == 0:
        print("未通过: jsonl 里有 image 字段，但当前候选目录下没有找到对应图片文件。")
        print("这通常说明官方仓库只包含样例 jsonl，没有包含表格图片本体。")
        print("下一步: 需要准备官方 table image 目录，或用 --image-root 指向你下载/上传后的图片目录。")
    else:
        print("部分通过: 有些图片能找到，有些图片缺失。")
        print("下一步: 检查缺失图片是否在其他目录，或补齐数据集图片文件。")


if __name__ == "__main__":
    main()
