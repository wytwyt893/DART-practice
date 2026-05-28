import torch
from torch.utils.data import Dataset


# ==========================
# Synthetic datasets for router sanity-check experiments
# ==========================


class SyntheticDataset(Dataset):
    """
    用于验证训练流程是否跑通的 toy 数据集。

    每个样本是一整块 feature，形状为 [feature_dim]。
    我们会向部分维度注入和 route label 有关的信号，
    让 MLP router 可以学到规律，而不是完全随机猜。
    """

    def __init__(
        self,
        num_samples: int = 1000,
        feature_dim: int = 10112,
        num_routes: int = 3,
    ):
        """
        Args:
            num_samples: 数据集总样本数。
            feature_dim: 每个样本的特征维度。
            num_routes: 路由数量，当前固定为 3。
        """
        if num_routes != 3:
            raise ValueError("This synthetic dataset currently expects num_routes=3.")

        self.features = torch.randn(num_samples, feature_dim) # 每个样本的 feature 是随机噪声构成的。
        self.route_labels = torch.randint(0, num_routes, (num_samples,), dtype=torch.long) # 每个样本随机分配一个 route label。

        informative_idx = torch.arange(0, feature_dim, 16)# 每隔 16 个维度注入一个 informative signal，总共 632 个维度会被注入信号。
        route_offsets = torch.tensor([-2.0, 0.0, 2.0])# 根据 route label 的不同，在 informative_idx 位置注入不同强度的信号，形成可学习的模式。

        self.features[:, informative_idx] += route_offsets[self.route_labels].unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.route_labels[idx]


class MultiSourceSyntheticDataset(Dataset):
    """
    更接近 TableDART gate 输入形式的 synthetic dataset。

    旧版 SyntheticDataset 返回一整块 feature：
        feature -> MLPRouter

    这个新版数据集返回三路 feature：
        text_feature + image_feature + question_feature -> concat -> MLPRouter

    这一步是在模拟官方 TableDART 里的：
        text_gate_feats + vlm_gate_feats + q_embeddings_for_gate
    """

    def __init__(
        self,
        num_samples: int = 1000, # 数据集总样本数。
        text_dim: int = 4096,    # 模拟 TextExpert gate feature 的维度
        image_dim: int = 4096,   # 模拟 VLMExpert gate feature 的维度
        question_dim: int = 1920,# 模拟 question embedding 的维度
        num_routes: int = 3,     # 路由数量，当前固定为 3，对应 Text / Image / Fusion
    ):
        """
        Args:
            num_samples: 数据集总样本数。
            text_dim: 模拟 TextExpert gate feature 的维度。
            image_dim: 模拟 VLMExpert gate feature 的维度。
            question_dim: 模拟 question embedding 的维度。
            num_routes: 路由数量，当前固定为 3，对应 Text / Image / Fusion。
        """
        if num_routes != 3:
            raise ValueError("MultiSourceSyntheticDataset currently expects num_routes=3.")


        self.text_features = torch.randn(num_samples, text_dim) # 模拟 TextExpert gate feature 的维度。
        self.image_features = torch.randn(num_samples, image_dim) # 模拟 VLMExpert gate feature 的维度。
        self.question_features = torch.randn(num_samples, question_dim) # 模拟 question embedding 的维度。

        # 先随机指定每个样本应该走哪条 route，再向对应来源注入更强的信号。
        # route 0: text signal stronger
        # route 1: image signal stronger
        # route 2: question/fusion signal stronger
        self.route_labels = torch.randint(0, num_routes, (num_samples,), dtype=torch.long) # 每个样本随机分配一个 route label。

        text_idx = torch.arange(0, min(text_dim, 512), 16) # 每隔 16 个维度注入一个 informative signal，总共最多 32 个维度会被注入信号（如果 feature_dim >= 512 的话）。
                                                           # 返回值是一个一维 tensor，包含了被选中注入信号的维度索引。比如，如果 text_dim=4096，那么 text_idx 就是 [0, 16, 32, ..., 496]，总共 32 个维度。
        image_idx = torch.arange(0, min(image_dim, 512), 16)
        question_idx = torch.arange(0, min(question_dim, 512), 16)

        # route 0 -> text_feature 的部分维度 +2.0
        # route 1 -> image_feature 的部分维度 +2.0
        # route 2 -> question_feature 的部分维度 +2.0
        # text_feature 更强 -> route 0
        # image_feature 更强 -> route 1
        # question_feature 更强 -> route 2 / fusion
        signal_strength = 2.0 # 注入的信号强度，数值越大越容易学会区分三条路由。可以调小看看训练效果变差的情况。
        text_rows = torch.where(self.route_labels == 0)[0] # 找到应该走 text route 的样本行索引
        image_rows = torch.where(self.route_labels == 1)[0]
        question_rows = torch.where(self.route_labels == 2)[0]

        # 使用 row/column 的显式索引写回原 tensor，避免链式索引产生临时副本。
        # route_label == 0 的样本：只增强 text_feature 的部分维度
        # route_label == 1 的样本：只增强 image_feature 的部分维度
        # route_label == 2 的样本：只增强 question_feature 的部分维度
        # text_rows[:, None] 负责选“哪些样本行”，text_idx 负责选“哪些特征列”，组合后就是给这些样本的这些维度统一注入 +2 信号。
        self.text_features[text_rows[:, None], text_idx] += signal_strength 
        self.image_features[image_rows[:, None], image_idx] += signal_strength
        self.question_features[question_rows[:, None], question_idx] += signal_strength

    def __len__(self) -> int:
        return len(self.route_labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        返回一个字典样本，模拟官方 dataloader 的复杂 batch 风格。

        DataLoader 默认会把多个 dict 样本自动合并成 dict batch：
            batch["text_feature"] -> [batch_size, text_dim]
            batch["image_feature"] -> [batch_size, image_dim]
            batch["question_feature"] -> [batch_size, question_dim]
            batch["route_label"] -> [batch_size]
        """
        return {
            "text_feature": self.text_features[idx],
            "image_feature": self.image_features[idx],
            "question_feature": self.question_features[idx],
            "route_label": self.route_labels[idx],
        }
