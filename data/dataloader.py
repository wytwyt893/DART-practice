import torch
from torch.utils.data import Dataset
#==================================一个用于训练流程验证的合成数据集======================================

class SyntheticDataset(Dataset):
    """
    一个只用于验证训练流程是否跑通的 toy 数据集。

    每个样本本来先是随机向量，
    然后我们手动往部分维度里注入“跟路由有关的信号”，
    这样模型就能学到东西，不会完全瞎猜。
    """

    def __init__(self, num_samples: int = 1000, feature_dim: int = 10112, num_routes: int = 3):
        """
        Args:
            num_samples: 数据集总样本数。
            feature_dim: 每个样本的特征维度。
            num_routes: 路由数。当前这个 toy 版本默认模拟三条路由。
        """
        if num_routes != 3:
            raise ValueError("This synthetic dataset currently expects num_routes=3.")

        # 第一步: 先生成纯随机噪声特征。
        # 张量形状是 [num_samples, feature_dim]。
        self.features = torch.randn(num_samples, feature_dim)

        # 第二步: 给每个样本随机分一个路由标签。
        # 张量形状是 [num_samples]。
        self.route_labels = torch.randint(0, num_routes, (num_samples,), dtype=torch.long)

        # 第三步: 挑出一部分“有信息的维度”。
        # 这里是每隔16维取一个。
        # 其他大多数维度仍然只是噪声。
        informative_idx = torch.arange(0, feature_dim, 16)

        # 第四步: 给不同路由定义不同的偏移量。
        # 路由 0 对应 -2.0
        # 路由 1 对应  0.0
        # 路由 2 对应 +2.0
        # 这样不同路由在这些 informative 维度上会出现可区分的模式。
        route_offsets = torch.tensor([-2.0, 0.0, 2.0])

        # 第五步: 把“路由偏移量”加到 informative 维度上。
        # route_offsets[self.route_labels] 的意思是:
        # 根据每个样本自己的标签，取出对应的偏移量。
        # unsqueeze(1) 是为了把形状从 [num_samples]
        # 变成 [num_samples, 1]，
        # 这样才能在后面广播到多个 informative 维度上。
        self.features[:, informative_idx] += route_offsets[self.route_labels].unsqueeze(1)

    def __len__(self) -> int:
        """
        返回数据集大小。
        DataLoader 会通过它知道整个数据集里有多少个样本。
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引取出一个样本。

        Args:
            idx: 要取的样本编号。

        Returns:
            返回一个二元组: (特征向量, 路由标签)。
        """
        return self.features[idx], self.route_labels[idx]
