"""
检查 Text Expert prompt 构造结果。

这个脚本属于 Stage 5.3A：复现官方 TableDART 的 Text Expert 输入格式。

为什么要写这个脚本？
之前我们已经接入了真实 question embedding：

    question text -> MiniLM -> question_features.pt

但 Text Expert 不是只看 question。
官方 TableDART 的 Text Expert 输入是：

    Markdown table + question + category-specific instruction

也就是官方 dataloader.py 里返回的：

    prompt_for_text_expert

在真正提取 text_features.pt 之前，我们必须先确认：
1. 表格能不能转成 Markdown；
2. 不同 category 能不能选择到正确 prompt 模板；
3. 最终 prompt_for_text_expert 是否符合官方输入格式；
4. prompt 长度是否大致合理。

本脚本只做“检查”，不训练模型，也不调用 TableGPT2。

运行方式：

    python scripts/inspect_text_prompts.py

默认读取：

    configs/router_real_format_cached_question_mock.yaml

输出内容：
    每个类别展示 1 条样本：
    - question_id
    - category
    - question
    - prompt 字符数
    - prompt 前若干字符

如果这一步确认无误，下一步才是：

    prompt_for_text_expert
    -> tokenizer / text encoder
    -> text_features.pt
"""

from collections import OrderedDict
from pathlib import Path
import sys
import locale


# 当我们用 `python scripts/inspect_text_prompts.py` 运行脚本时，
# Python 默认会把 scripts/ 当作 import 起点。
# 但我们的 data/、utils/ 在项目根目录下，所以这里手动把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from data.real_format_dataloader import RealFormatMockFeatureDataset
from utils.config import load_config
from utils.seed import set_seed
from utils.text_prompt_builder import build_text_expert_prompt


def print_section(title: str) -> None:
    """打印分隔标题，让终端输出更容易阅读。"""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def shorten_text(text: str, max_chars: int) -> str:
    """
    截断过长 prompt，避免终端输出太爆炸。

    Text Expert prompt 往往包含整张 Markdown table，可能很长。
    检查脚本只需要看开头结构是否正确，所以默认只展示前 max_chars 个字符。
    """
    # Windows PowerShell 有时使用 GBK 编码，遇到 \xa0 这类特殊空格会打印失败。
    # 这里做展示层清洗：只影响终端检查输出，不改变真正构造 prompt 的逻辑。
    text = text.replace("\xa0", " ")

    if len(text) <= max_chars:
        display_text = text
    else:
        display_text = text[:max_chars] + "\n...[Prompt 过长，已截断展示]"

    # 终端展示层兜底：
    # 有些表格里会出现 ø、é、特殊货币符号等字符。
    # Windows PowerShell 如果当前不是 UTF-8 编码，直接 print 可能报 UnicodeEncodeError。
    # 这里按当前终端编码做一次 errors="replace" 的安全转换。
    console_encoding = locale.getpreferredencoding(False)
    return display_text.encode(console_encoding, errors="replace").decode(
        console_encoding
    )


def main() -> None:
    # 1. 读取 Stage 5.2/5.3 当前主线配置。
    #
    # 这里用 cached question 配置，是因为它代表当前最新复现主线：
    # - 官方真实 jsonl
    # - 论文维度
    # - cached question feature
    #
    # 但本脚本不关心 question_features.pt，本脚本只是复用同一批样本顺序。
    config_path = PROJECT_ROOT / "configs" / "router_real_format_cached_question_mock.yaml"
    config = load_config(config_path)

    print_section("1. 配置信息")
    print(f"实验名称: {config['experiment_name']}")
    print(f"数据路径: {config['data']['data_path']}")
    print(f"每类样本数: {config['data'].get('samples_per_category')}")

    set_seed(config["seed"])

    # 2. 构造 Dataset。
    #
    # 注意：
    # 本脚本并不使用模型输入 feature 来训练，只需要 dataset.samples 里的原始字段：
    # - table
    # - question
    # - category
    # - question_id
    #
    # 但为了保证和训练数据顺序一致，我们仍然复用 RealFormatMockFeatureDataset 的采样逻辑。
    dataset = RealFormatMockFeatureDataset(
        data_path=config["data"]["data_path"],
        max_samples=config["data"]["max_samples"],
        samples_per_category=config["data"].get("samples_per_category"),
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
        use_cached_question_features=config["data"].get(
            "use_cached_question_features", False
        ),
        question_feature_path=config["data"].get("question_feature_path"),
    )

    print_section("2. 数据集概览")
    print(f"样本总数: {len(dataset.samples)}")

    # 3. 每个 category 取 1 条样本展示。
    #
    # 用 OrderedDict 是为了保留数据出现顺序。
    # 这样输出顺序和 dataset.samples 的顺序一致，更容易和前面检查脚本对应。
    examples_by_category = OrderedDict()
    for item in dataset.samples:
        category = item.get("category", "")
        if category not in examples_by_category:
            examples_by_category[category] = item

    print(f"覆盖类别数: {len(examples_by_category)}")
    for category in examples_by_category:
        print(f"  - {category}")

    # 4. 构造并展示 prompt。
    #
    # 这里调用的 build_text_expert_prompt 是 Stage 5.3A 的核心函数：
    # table + question + category -> prompt_for_text_expert
    print_section("3. 文本专家 Prompt 示例")
    for category, item in examples_by_category.items():
        table = item.get("table", {})
        question = item.get("question", "")
        question_id = item.get("question_id", "")

        prompt = build_text_expert_prompt(
            table=table,
            question=question,
            category=category,
        )

        print("\n" + "-" * 88)
        print(f"问题 ID: {question_id}")
        print(f"任务类别: {category}")
        print(f"原始问题: {question}")
        print(f"Prompt 字符数: {len(prompt)} 字符")
        print("-" * 88)
        print(shorten_text(prompt, max_chars=1200))

    print_section("4. 检查结论")
    print("通过: 已能从真实表格、问题和任务类别构造文本专家 Prompt。")
    print("下一步: 使用这些 Prompt 提取 text_features.pt。")


if __name__ == "__main__":
    main()
