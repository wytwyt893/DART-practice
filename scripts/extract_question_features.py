import json
from pathlib import Path
import sys

import torch

# 根目录强化，确保无论从哪个子目录运行脚本，都能正确导入项目里的模块。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_format_dataloader import RealFormatMockFeatureDataset
from utils.config import load_config
from utils.question_embedding import get_question_embedding_for_gate
from utils.seed import set_seed

def main() -> None:
    # 1. 加载配置文件，获取数据参数。
    config_path = PROJECT_ROOT / "configs" / "router_real_format_real_question_mock.yaml"
    config = load_config(config_path)

    print(f"Loaded config: {config['experiment_name']}")

    # 2. 设置随机种子，确保结果可复现。
    set_seed(config["seed"])

    # 3. 创建数据集和数据加载器。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. 构造数据集时，直接生成问题嵌入特征，而不是随机特征。
    dataset = RealFormatMockFeatureDataset(
        data_path=config["data"]["data_path"],
        max_samples=config["data"]["max_samples"],
        samples_per_category=config["data"].get("samples_per_category"),
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
        use_real_question_embedding=False,
    )

    # 5. 划分训练集和验证集，创建对应的 DataLoader。
    output_dir = PROJECT_ROOT / "data" / "features" / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "question_features.pt"
    metadata_path = output_dir / "metadata.json"

    # 6. 遍历整个数据集，提取每个问题的嵌入特征，并保存到列表中
    # 同时收集一些元信息（如 question_id、category）以便后续分析
    question_features = []
    metadata = []

    # 7. 这里直接遍历 dataset.samples 列表，而不是通过 DataLoader，
    for idx, item in enumerate(dataset.samples):
        question = item.get("question", "")
        question_id = item.get("question_id", f"unknown_{idx}")
        category = item.get("category", "")

        # 这是embedding的核心函数，输入是问题文本，输出是一个向量特征。
        question_embedding = get_question_embedding_for_gate(
            question_text=question,
            device=device,
            model_id=config["data"].get(
                "question_embed_model_id",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )

        # 把embedding添加到列表中，同时保存对应的元信息
        question_features.append(question_embedding)

        metadata.append(
            {
                "feature_index": idx,
                "question_id": question_id,
                "category": category,
                "question": question,
            }
        )

        # 每提取 50 个特征，就打印一次进度，方便监控长时间运行的过程。
        if (idx + 1) % 50 == 0:
            print(f"Extracted {idx + 1}/{len(dataset)} question features")

    # 8. 把所有的特征堆叠成一个大的 tensor，并保存到磁盘上。
    question_features = torch.stack(question_features)

    if question_features.shape[1] != config["data"]["question_dim"]:
        raise ValueError(
            f"Question feature dim mismatch: "
            f"got {question_features.shape[1]}, expected {config['data']['question_dim']}"
        )

    # 9. 同时把元信息保存成一个 JSON 文件，方便后续分析和验证。
    torch.save(question_features, feature_path)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved question features to: {feature_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Feature shape: {question_features.shape}")

# if __name__ == "__main__" 这个条件判断的作用是确保当这个脚本被直接运行时
# 才会执行 main() 函数
# 这样做的好处是，如果这个脚本被其他脚本导入作为模块使用时
# main() 函数不会自动执行，而是需要显式调用 main() 才会运行
if __name__ == "__main__":
    main()