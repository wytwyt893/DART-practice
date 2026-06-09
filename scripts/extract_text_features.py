"""
Stage 5.3B: 离线提取真实 Text Expert gate feature。

这个脚本解决的问题：
    我们之前已经完成了 Stage 5.3A：

        table / question / category
        -> Markdown table
        -> Text Expert Prompt

    也就是构造出了官方 TableDART 里会送给 Text Expert 的
    prompt_for_text_expert。

    但是，到目前为止，训练用的 text_feature 仍然是 mock 随机特征。
    如果想进一步接近真实 TableDART 复现，就要把 text_feature 换成
    真实文本专家模型产生的 gate feature。

这个脚本做什么：
    对每条样本执行下面的数据流：

        官方 jsonl 样本
        -> table / question / category
        -> build_text_expert_prompt(...)
        -> TableGPT2 tokenizer
        -> input_ids / attention_mask
        -> TableGPT2 input embedding
        -> masked mean pooling
        -> text feature, 形状通常是 [3584]

    最后把所有样本的 text feature 堆叠成：

        text_features.pt, 形状 [num_samples, 3584]

为什么要“离线”提取：
    TableGPT2-7B 很大，加载和计算成本都比我们之前的 MiniLM 高很多。
    如果在 train loop 的每个 epoch 都调用 TableGPT2，会非常慢，也更容易 OOM。

    所以科研复现里常见做法是：

        先离线提取大模型特征并缓存
        -> 再用缓存特征训练轻量 router / selector

    这和我们之前缓存 question_features.pt 是同一个思路。

和官方代码的对应关系：
    官方 TableDART 的 TableGPT2Expert.extract_gate_features_and_intermediate_state
    核心逻辑可以简化为：

        tokens = tokenizer(prompts_batch, padding=True, truncation=True, ...)
        embeds = model.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1)
        pooled = (embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    本脚本复现的就是这段 gate feature 提取逻辑。

注意：
    1. 这个脚本会加载 tablegpt/TableGPT2-7B，建议在 AutoDL 或服务器上运行。
    2. 本地 RTX 4060 Laptop GPU 不建议直接跑完整提取。
    3. 第一次运行会从 Hugging Face 下载模型权重，需要网络、磁盘空间和时间。
    4. 这个脚本只提取 gate feature，不生成答案，也不训练模型。

推荐先在 AutoDL 上做 smoke test：

    python scripts/extract_text_features.py --max-samples 8 --batch-size 1

确认能下载模型、能加载模型、能保存 feature 后，再跑完整数据：

    python scripts/extract_text_features.py --batch-size 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 当我们用 `python scripts/extract_text_features.py` 运行脚本时，
# Python 会把 scripts/ 当成 import 起点。
# 但项目模块 data/、utils/ 都在项目根目录下，
# 所以这里手动把项目根目录加入 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from data.real_format_dataloader import RealFormatMockFeatureDataset
from utils.config import load_config
from utils.seed import set_seed
from utils.text_prompt_builder import build_text_expert_prompt


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "router_real_format_cached_question_mock.yaml"
DEFAULT_TEXT_MODEL_ID = "tablegpt/TableGPT2-7B"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    为什么要给脚本加命令行参数？
        因为真实大模型提取特征时，我们经常需要分阶段跑：

        - 本地 / 云端 smoke test：只跑 8 条样本；
        - 正式提取：跑完整数据；
        - 显存不够：把 batch_size 调成 1；
        - 模型已提前下载：使用 local_files_only 避免重复联网。

    有了 argparse，就不用每次都改源码，只需要改运行命令。
    """
    parser = argparse.ArgumentParser(
        description="Extract real TableGPT2 text gate features for TableDART reproduction."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="实验配置文件路径。默认读取当前 Stage 5.2/5.3 主线配置。",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_TEXT_MODEL_ID,
        help="Text Expert 模型 ID。默认使用官方配置中的 tablegpt/TableGPT2-7B。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "输出目录。默认保存到 data/features/<experiment_name>/。"
            "脚本会在该目录下生成 text_features.pt 和 text_metadata.json。"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "最多提取多少条样本。建议第一次上 AutoDL 时设置为 8 做 smoke test。"
            "如果不填，就使用 config 里的 max_samples / samples_per_category 设置。"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="特征提取 batch size。7B 模型建议先用 1，确认显存充足后再调大。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "tokenizer 最大输入长度。不填则根据 tokenizer.model_max_length 自动推断，"
            "并预留 reserve_new_tokens。"
        ),
    )
    parser.add_argument(
        "--reserve-new-tokens",
        type=int,
        default=64,
        help=(
            "给后续生成预留的 token 数。官方代码会用 model_max_length 减去生成长度，"
            "这里虽然不生成答案，但保持同样截断习惯。"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型加载精度。AutoDL GPU 上通常 auto/float16 更省显存。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备。默认 auto：有 CUDA 就用 cuda，否则用 cpu。",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只从本地 Hugging Face 缓存加载模型，不联网下载。",
    )

    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    """
    根据命令行参数选择运行设备。

    - auto：优先用 cuda；
    - cuda：强制用 GPU，如果没有 GPU 会报错；
    - cpu：强制用 CPU，一般只适合语法/小样本调试，不适合 7B 完整提取。
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False。")

    return device


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """
    选择模型权重加载精度。

    为什么要关心 dtype？
        7B 模型如果用 float32 加载，大约会占用非常多显存/内存。
        在 GPU 上一般使用 float16 或 bfloat16 更合理。

    auto 的规则：
        - CPU：float32，兼容性最好；
        - CUDA 且支持 bfloat16：bfloat16；
        - CUDA 但不支持 bfloat16：float16。
    """
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    if device.type == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_tokenizer_max_length(
    tokenizer: Any,
    max_length_arg: int | None,
    reserve_new_tokens: int,
) -> int:
    """
    决定 tokenizer 的最大输入长度。

    官方 Text Expert 会做 truncation=True，避免超长表格 prompt 把模型输入撑爆。
    它大致使用：

        tokenizer.model_max_length - MAX_NEW_TOKENS

    但 Hugging Face 里有些 tokenizer 的 model_max_length 会是一个特别巨大的数，
    表示“没有明确限制”。这种情况下我们手动回退到 4096。
    """
    if max_length_arg is not None:
        return max_length_arg

    model_max_length = getattr(tokenizer, "model_max_length", None)

    if not isinstance(model_max_length, int) or model_max_length > 100_000:
        model_max_length = 4096

    return max(1, model_max_length - reserve_new_tokens)


def load_text_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[Any, Any]:
    """
    加载 TableGPT2 tokenizer 和模型。

    这里我们不直接 import 官方 TableGPT2Expert，原因是：
        1. 官方 experts.py 顶部会做 Hugging Face login、导入很多多模态依赖；
        2. 我们当前只需要 Text Expert 的 gate feature 提取逻辑；
        3. 直接用 Transformers 加载 tokenizer/model 更清晰，也更容易 debug。

    注意：
        即使我们只用 input embedding，AutoModelForCausalLM.from_pretrained
        仍然会加载完整模型权重。
    """
    print(f"正在加载 tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    # GPT 类模型有时没有 pad_token。
    # 但 batch tokenizer 需要 padding，所以官方代码会把 pad_token 设置成 eos_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 官方 TableGPT2Expert 使用 left padding。
    # 对 decoder-only LM 来说，left padding 通常更适合批量生成场景。
    # 虽然我们这里只做 embedding pooling，仍然保持和官方一致。
    tokenizer.padding_side = "left"

    print(f"正在加载 Text Expert 模型: {model_id}")
    print(f"模型 dtype: {dtype}")
    print(f"目标设备: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    return tokenizer, model


def build_dataset_from_config(config: dict, max_samples_override: int | None) -> RealFormatMockFeatureDataset:
    """
    根据 config 构造 Dataset。

    为什么还用 RealFormatMockFeatureDataset？
        因为它已经实现了：
        - 读取官方 jsonl；
        - 按 samples_per_category 做均衡采样；
        - 保留 question_id / category / table / question 等原始字段。

    本脚本并不使用 dataset 里生成的 mock text_feature。
    我们只复用 dataset.samples 的样本顺序和真实字段。

    这点非常关键：
        离线提取出来的 text_features.pt 第 i 行，
        必须对应 dataset.samples 第 i 条样本。

    如果顺序错了，后续 router 训练时 feature 和 label 就会错位。
    """
    data_cfg = config["data"]

    max_samples = max_samples_override
    if max_samples is None:
        max_samples = data_cfg.get("max_samples")

    dataset = RealFormatMockFeatureDataset(
        data_path=data_cfg["data_path"],
        max_samples=max_samples,
        samples_per_category=data_cfg.get("samples_per_category"),
        text_dim=data_cfg["text_dim"],
        image_dim=data_cfg["image_dim"],
        question_dim=data_cfg["question_dim"],
        num_routes=data_cfg["num_routes"],
        use_real_question_embedding=False,
        use_cached_question_features=False,
    )

    return dataset


def build_prompts_and_metadata(dataset: RealFormatMockFeatureDataset) -> tuple[list[str], list[dict]]:
    """
    把 dataset.samples 转换成 Text Expert prompts，并同步生成 metadata。

    prompts:
        真实要送进 tokenizer / TableGPT2 的字符串列表。

    metadata:
        用来记录每个 feature 对应哪条样本，方便后续排查。
        比如你以后发现第 17 条特征异常，可以从 metadata 里找回 question_id。
    """
    prompts: list[str] = []
    metadata: list[dict] = []

    for idx, item in enumerate(dataset.samples):
        table = item.get("table", {})
        question = item.get("question", "")
        category = item.get("category", "")
        question_id = item.get("question_id", f"unknown_{idx}")

        prompt = build_text_expert_prompt(
            table=table,
            question=question,
            category=category,
        )

        prompts.append(prompt)
        metadata.append(
            {
                "feature_index": idx,
                "question_id": question_id,
                "category": category,
                "question": question,
                "prompt_char_length": len(prompt),
                # 只保存前 300 个字符作为预览，避免 metadata.json 过大。
                "prompt_preview": prompt[:300],
            }
        )

    return prompts, metadata


@torch.no_grad()
def extract_text_features(
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    逐 batch 提取 Text Expert gate feature。

    核心张量形状：
        prompts_batch:
            Python list[str]，长度 batch_size。

        tokens.input_ids:
            [batch_size, seq_len]
            每个 token 的离散编号。

        tokens.attention_mask:
            [batch_size, seq_len]
            1 表示真实 token，0 表示 padding。

        embeds:
            [batch_size, seq_len, hidden_size]
            input_ids 经过 TableGPT2 输入 embedding 层之后得到的连续向量。

        mask:
            [batch_size, seq_len, 1]
            为了和 embeds 相乘，把 attention_mask 扩展一个维度。

        pooled:
            [batch_size, hidden_size]
            对真实 token 的 embedding 做平均池化后得到的整条 prompt 表示。

    pooled 就是我们要保存的 text_feature。
    """
    all_features: list[torch.Tensor] = []
    total = len(prompts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        prompts_batch = prompts[start:end]

        tokens = tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        # input_ids 是离散 token id。
        # get_input_embeddings() 是模型的 embedding 层，
        # 它把 token id 映射成连续向量。
        embeds = model.get_input_embeddings()(tokens["input_ids"])

        # attention_mask 中 1 表示真实 token，0 表示 padding。
        # unsqueeze(-1) 后形状从 [B, L] 变成 [B, L, 1]，
        # 这样才能和 embeds 的 [B, L, H] 做广播相乘。
        mask = tokens["attention_mask"].unsqueeze(-1)

        # masked mean pooling:
        # 先把 padding 位置的 embedding 乘成 0，
        # 再除以真实 token 数量，得到每条 prompt 的整体向量表示。
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 训练 router 时我们只需要 CPU 上的缓存 tensor。
        # 所以这里 detach + float + cpu：
        # - detach: 不保留计算图；
        # - float: 统一保存成 float32，后续训练更稳定；
        # - cpu: 避免一直占用 GPU 显存。
        all_features.append(pooled.detach().float().cpu())

        print(f"已提取 {end}/{total} 条 text features")

    return torch.cat(all_features, dim=0)


def save_outputs(
    text_features: torch.Tensor,
    metadata: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    保存特征和 metadata。

    text_features.pt:
        训练/评估脚本后续真正读取的 tensor 文件。

    text_metadata.json:
        给人检查用，记录 feature_index 和 question_id/category 的对应关系。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "text_features.pt"
    metadata_path = output_dir / "text_metadata.json"

    torch.save(text_features, feature_path)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return feature_path, metadata_path


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    print(f"Loaded config: {config['experiment_name']}")

    set_seed(config["seed"])

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    dataset = build_dataset_from_config(
        config=config,
        max_samples_override=args.max_samples,
    )
    print(f"成功读取样本数: {len(dataset.samples)}")

    prompts, metadata = build_prompts_and_metadata(dataset)
    print(f"成功构造 Text Expert Prompt 数量: {len(prompts)}")

    tokenizer, model = load_text_model(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )

    max_length = resolve_tokenizer_max_length(
        tokenizer=tokenizer,
        max_length_arg=args.max_length,
        reserve_new_tokens=args.reserve_new_tokens,
    )
    print(f"tokenizer max_length: {max_length}")

    text_features = extract_text_features(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=max_length,
    )

    expected_text_dim = config["data"]["text_dim"]
    if text_features.shape[1] != expected_text_dim:
        raise ValueError(
            f"Text feature dim mismatch: got {text_features.shape[1]}, "
            f"expected {expected_text_dim}. "
            "如果这里报错，说明当前加载的 Text Expert hidden_size "
            "和 config/论文中的 text_dim 不一致。"
        )

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "features" / config["experiment_name"]
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    feature_path, metadata_path = save_outputs(
        text_features=text_features,
        metadata=metadata,
        output_dir=output_dir,
    )

    print("\n提取完成。")
    print(f"Text feature 保存路径: {feature_path}")
    print(f"Metadata 保存路径: {metadata_path}")
    print(f"Feature shape: {text_features.shape}")


if __name__ == "__main__":
    main()
