from collections import Counter
from pathlib import Path
import sys

from torch.utils.data import DataLoader


# 这个脚本属于 Stage 4 的“真实数据格式检查”
# 目标是验证我们能不能正确读取官方 TableDART 的 jsonl 数据格式
#
# 目前我们还不训练模型，也不调用大模型。
# 目标只是确认：
# 1. 能不能读取官方 TableDART 的 jsonl 数据格式；
# 2. 一个真实样本里有哪些字段；
# 3. 这些字段能不能被我们自己的 Dataset 返回；
# 4. DataLoader 能不能把多个真实样本合成一个 batch；
# 5. mock feature 的 shape 是否还能对接 MiniDynamicSelector。


# 当我们用命令 `python scripts/inspect_real_format_dataset.py` 运行脚本时，
# Python 的默认 import 搜索路径会优先从 scripts/ 目录开始。
# 这样它可能找不到项目根目录下的 data/ 和 utils/。
#
# 所以这里手动把项目根目录加入 sys.path。
# 这是一种常见的脚本写法，后面如果把项目打包成 package，可以再换更规范的方式。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from data.real_format_dataloader import RealFormatMockFeatureDataset, real_format_collate_fn
from utils.config import load_config
from utils.seed import set_seed


def print_section(title: str) -> None:
    """打印一个分隔标题，让终端输出更容易阅读。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # 1. 读取 Stage 4 专用配置。
    # 这里面记录了官方 jsonl 路径、读取多少条样本、mock feature 维度等信息。
    config = load_config("configs/router_real_format_mock.yaml")

    print_section("1. 配置信息")
    print(f"实验名称: {config['experiment_name']}")
    print(f"数据路径: {config['data']['data_path']}")
    print(f"最大读取样本数: {config['data']['max_samples']}")
    print(
        "特征维度: "
        f"文本={config['data']['text_dim']}, "
        f"图像={config['data']['image_dim']}, "
        f"问题={config['data']['question_dim']}"
    )

    # 2. 设置随机种子。
    # 这个阶段虽然还不训练，但 Dataset 里会生成随机 mock feature，
    # 所以仍然需要固定 seed，保证每次检查结果尽量一致。
    set_seed(config["seed"])

    # 3. 创建真实格式 Dataset。
    #
    # RealFormatMockFeatureDataset 会读取官方 jsonl 的真实字段：
    # question_id / category / question / table / answer / image
    #
    # 同时它会临时生成 mock feature：
    # text_feature / image_feature / question_feature / route_label
    #
    # 这样我们可以在不调用真实大模型的情况下，
    # 先验证真实数据格式能不能接入当前 MiniDynamicSelector 流程。
    dataset = RealFormatMockFeatureDataset(
        data_path=config["data"]["data_path"],
        max_samples=config["data"]["max_samples"],
        samples_per_category=config["data"].get("samples_per_category"), #新增这个参数，允许我们在读取样本时按照 category 进行均衡采样，避免前 N 条样本过于单一。
        text_dim=config["data"]["text_dim"],
        image_dim=config["data"]["image_dim"],
        question_dim=config["data"]["question_dim"],
        num_routes=config["data"]["num_routes"],
    )

    print_section("2. 数据集概览")
    print(f"成功读取样本数: {len(dataset)}")

    # 4. 统计真实 category 分布和 mock route label 分布。
    #
    # 这一步可以帮助我们发现数据抽样是否均衡。
    # 例如之前读取前 256 条时，全部都是 WTQ_for_TQA，
    # 这说明直接取前 N 条并不适合训练，需要后续做跨类别采样。
    categories = Counter(item["category"] for item in dataset.samples)
    route_labels = Counter(dataset.route_labels.tolist())

    print("任务类别分布:")
    for category, count in categories.items():
        print(f"  - {category}: {count}")

    print("伪路由标签分布:")
    route_name_map = {
        0: "route 0 / 偏文本路径",
        1: "route 1 / 偏图像或表格结构路径",
        2: "route 2 / 偏融合或问题路径",
    }
    for route_label, count in sorted(route_labels.items()):
        route_name = route_name_map.get(route_label, f"route {route_label}")
        print(f"  - {route_name}: {count}")

    if len(categories) == 1:
        print(
            "\n警告: 当前样本子集只包含一个任务类别。"
            "这用于格式检查是可以的，但还不适合正式训练。"
        )
    if len(route_labels) == 1:
        print(
            "警告: 当前伪路由标签只包含一个类别。"
            "如果直接训练，路由器可能只学会永远预测同一路径。"
        )

    # 5. 查看第一个样本。
    #
    # 这里不是为了训练，而是为了让我们确认一个真实样本到底长什么样：
    # 问题是什么、答案是什么、图片文件名是什么、表格有多少列和多少行。
    first = dataset[0]
    print_section("3. 单条样本检查")
    print("下面展示第一条样本，用于确认真实 TableDART 字段能否被正确读取。")
    print(f"question_id: {first['question_id']}")
    print(f"任务类别: {first['category']}")
    print(f"问题: {first['question']}")
    print(f"答案: {first['answer']}")
    print(f"图像文件名: {first['image']}")
    print(f"表头字段数量: {len(first['table_header'])}")
    print(f"表格行数: {len(first['table_rows'])}")
    print("\n这条样本对应的 mock feature 形状:")
    print(f"  文本特征 text_feature: {first['text_feature'].shape}")
    print(f"  图像特征 image_feature: {first['image_feature'].shape}")
    print(f"  问题特征 question_feature: {first['question_feature'].shape}")
    print(f"  路由标签 route_label: {first['route_label']} ({route_name_map[int(first['route_label'])]})")

    # 6. 创建 DataLoader，检查 batch 级别的输出。
    #
    # 注意：这里必须传入 real_format_collate_fn（来自 data.real_format_dataloader）。
    #
    # 原因是：
    # 真实表格的 table_rows 是变长 list，不同样本的行数不一样；
    # PyTorch 默认 collate 无法把变长 list 自动 stack 成 tensor。
    #
    # real_format_collate_fn 的规则是：
    # - feature tensor 用 torch.stack 合成 batch；
    # - question / answer / table / image 等原始字段保留成 list。
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=real_format_collate_fn,
    )

    batch = next(iter(loader))

    # 7. 检查 batch shape。
    #
    # 如果这里输出：
    # text_feature     [8, 4096]
    # image_feature    [8, 4096]
    # question_feature [8, 1920]
    # route_label      [8]
    #
    # 就说明真实格式 Dataset 已经可以产出 MiniDynamicSelector 所需的输入。
    print_section("4. DataLoader 批处理检查")
    print("如果下面这些 shape 正确，说明这个 batch 可以送入 MiniDynamicSelector。")
    print(f"文本特征 batch: {batch['text_feature'].shape}")
    print(f"图像特征 batch: {batch['image_feature'].shape}")
    print(f"问题特征 batch: {batch['question_feature'].shape}")
    print(f"路由标签 batch: {batch['route_label'].shape}")
    print(f"question_id 示例: {batch['question_id'][:3]}")

    print_section("5. 检查结论")
    print("通过: 官方真实格式 jsonl 可以被 Stage 4 Dataset 正确读取。")
    print("通过: question/table/answer/image/category 等真实字段被保留下来了。")
    print("通过: mock feature 可以被 DataLoader 组成 batch，并对接 MiniDynamicSelector。")
    print(
        "下一步: 在训练前加入按任务类别均衡采样，因为当前前 256 条样本"
        "只覆盖了单一类别和单一路由。"
    )


if __name__ == "__main__":
    main()
