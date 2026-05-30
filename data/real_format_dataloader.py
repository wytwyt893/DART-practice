import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RealFormatMockFeatureDataset(Dataset):
    """
    Stage 4: 读取真实 TableDART jsonl 格式，但暂时仍使用 mock feature。

    这一步的目标不是调用真实大模型，而是先让我们的工程能处理真实字段：
        question_id / category / question / table / answer / image

    同时，为了复用 Stage 2 的 MiniDynamicSelector，我们仍然返回：
        text_feature / image_feature / question_feature / route_label

    注意：
    route_label 目前仍是 mock 标签，用来验证训练流程；
    它还不是官方 TableDART 根据专家答案正确性构造的 soft target。
    """

    def __init__(
        self,
        data_path: str, 
        max_samples: int | None = None, #意思是如果 max_samples 是 None，就加载所有样本；否则只加载前 max_samples 个样本。
        samples_per_category: int | None = None,
        text_dim: int = 4096, #文本特征维度，保持和 Stage 2 一致，方便后续替换成真实大模型输出。
        image_dim: int = 4096,
        question_dim: int = 1920,
        num_routes: int = 3,
    ):
        if num_routes != 3: #加判断，因为目前 mock route label 的构造规则是固定的，针对 3 条路由设计的。
            raise ValueError("RealFormatMockFeatureDataset currently expects num_routes=3.")

        self.data_path = Path(data_path) #使用 Path 对象更方便地处理文件路径。
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.question_dim = question_dim
        self.num_routes = num_routes

        if not self.data_path.exists(): #加个文件存在的检查，避免后续打开文件时报错不清晰。
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        all_samples = [] #首先读取所有样本到内存中，方便后续根据 max_samples 和 samples_per_category 进行筛选。
        with self.data_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                all_samples.append(item)

        if samples_per_category is not None: #如果指定了 samples_per_category，就按照 category 进行分组抽样，保证每个类别的样本数不超过 samples_per_category。
            category_counts = {}
            self.samples = []

            for item in all_samples: #遍历所有样本，根据 category 字段统计每个类别已经抽取的样本数，如果当前类别的样本数还没有达到 samples_per_category，就将该样本加入最终的 samples 列表中。
                category = item.get("category", "")
                current_count = category_counts.get(category, 0)

                if current_count < samples_per_category: #如果当前类别的样本数还没有达到 samples_per_category，就将该样本加入最终的 samples 列表中。
                    self.samples.append(item)
                    category_counts[category] = current_count + 1
        else:
            self.samples = all_samples

        if max_samples is not None: #如果指定了 max_samples，就直接取前 max_samples 个样本。这个操作是在 samples_per_category 之后进行的，所以它是全局的样本数限制。
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise ValueError(f"No samples loaded from {self.data_path}")

        self.text_features = torch.randn(len(self.samples), text_dim)#为每个样本生成随机的文本、图像和问题特征。这些特征目前是随机的，主要目的是为了让数据结构完整，能够通过后续的训练流程。
        self.image_features = torch.randn(len(self.samples), image_dim)
        self.question_features = torch.randn(len(self.samples), question_dim)

        # Stage 4 暂时用 category 构造 mock route label。
        # 这个规则只是为了让 real-format 数据也能跑通 router 训练闭环。
        self.route_labels = torch.tensor(
            [self._build_mock_route_label(item) for item in self.samples],
            dtype=torch.long,
        )

        self._inject_mock_signal()

    def _build_mock_route_label(self, item: dict) -> int:
        """
        根据真实样本的 category 构造一个临时 route label。

        当前只是 mock 规则：
        - TFV 类任务更偏文本语义判断 -> route 0
        - TABMWP / WTQ 等结构或数值查询 -> route 1
        - 生成式或复杂表格任务 -> route 2
        """
        category = item.get("category", "")

        # 判断原理为：根据 TableDART 的任务分类
        # TFV、TabFact 和 InfoTabs 主要关注文本理解和事实验证，因此我们将它们归为 route 0；
        # 而 TABMWP 和 WTQ 则更侧重于表格结构查询或数值计算，因此归为 route 1；
        # 其他类型的任务（如生成式任务）则归为 route 2。
        # 这只是一个简单的 mock 规则，实际情况可能更复杂，但足以让我们验证训练流程的正确性。
        if "TFV" in category or "TabFact" in category or "InfoTabs" in category: #这些类别通常涉及文本理解和事实验证，更适合走文本语义判断的路由。
            return 0
        if "TABMWP" in category or "WTQ" in category: #这些类别通常涉及表格结构查询或数值计算，更适合走结构化数据处理的路由。
            return 1
        return 2

    def _inject_mock_signal(self) -> None:
        """向 route 对应的 feature source 注入可学习信号。"""
        text_idx = torch.arange(0, min(self.text_dim, 512), 16)
        image_idx = torch.arange(0, min(self.image_dim, 512), 16)
        question_idx = torch.arange(0, min(self.question_dim, 512), 16)

        signal_strength = 2.0
        text_rows = torch.where(self.route_labels == 0)[0] #根据 route label 的值找到对应的样本行索引。例如，route label 为 0 的样本行索引存储在 text_rows 中，表示这些样本更适合走文本语义判断的路由。
        image_rows = torch.where(self.route_labels == 1)[0]
        question_rows = torch.where(self.route_labels == 2)[0]

        self.text_features[text_rows[:, None], text_idx] += signal_strength #在文本特征中，针对 route label 为 0 的样本行（text_rows），在指定的特征维度索引（text_idx）上注入一个强信号（signal_strength）。
        self.image_features[image_rows[:, None], image_idx] += signal_strength #在图像特征中，针对 route label 为 1 的样本行（image_rows），在指定的特征维度索引（image_idx）上注入一个强信号（signal_strength）。
        self.question_features[question_rows[:, None], question_idx] += signal_strength

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        table = item.get("table", {})

        return { #返回格式包含了文本特征、图像特征、问题特征、路由标签，以及原始的 question_id、category、question、answer、image 和 table 等字段。
                 #这些字段的设计是为了让后续的训练流程能够同时利用特征信息和原始数据内容。
            "text_feature": self.text_features[idx],
            "image_feature": self.image_features[idx],
            "question_feature": self.question_features[idx],
            "route_label": self.route_labels[idx],
            "question_id": item.get("question_id", f"unknown_{idx}"),
            "category": item.get("category", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "image": item.get("image", ""),
            "table": table,
            "table_header": table.get("header", []),
            "table_rows": table.get("rows", []),
            "table_name": table.get("name", ""),
        }


# 函数作用是定义一个自定义的 collate 函数
# 用于在 DataLoader 中将多个样本合并成一个 batch。
# 由于真实 table 字段是变长结构，不能直接使用 PyTorch 默认的 collate 函数进行堆叠，因此我们需要根据字段类型进行不同的处理：
def real_format_collate_fn(batch: list[dict]) -> dict:
    """
    真实 table 字段是变长结构，不能用 PyTorch 默认 collate 直接堆叠。

    处理规则：
    - tensor 字段：stack 成 batch tensor
    - 文本/表格/图片等原始字段：保留成 list
    """
    return {
        "text_feature": torch.stack([item["text_feature"] for item in batch]),
        "image_feature": torch.stack([item["image_feature"] for item in batch]),
        "question_feature": torch.stack([item["question_feature"] for item in batch]),
        "route_label": torch.stack([item["route_label"] for item in batch]),
        "question_id": [item["question_id"] for item in batch],
        "category": [item["category"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "image": [item["image"] for item in batch],
        "table": [item["table"] for item in batch],
        "table_header": [item["table_header"] for item in batch],
        "table_rows": [item["table_rows"] for item in batch],
        "table_name": [item["table_name"] for item in batch],
    }
