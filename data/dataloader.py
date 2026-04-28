import torch
from torch.utils.data import Dataset,DataLoader # 这个 Dataset 是 PyTorch 定义的一个抽象类，我们要继承它来创建自己的数据集；DataLoader 是一个工具类，能帮我们把 Dataset 包装成一个可迭代的对象，方便我们在训练循环中使用。

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, feature_dim=10112, num_classes=3): 
        """
        第一板斧：初始化。
        在这里把整体的“假特征”和“假标签”生成出来，存为类的属性（比如 self.features）。
        """
        # 1. 生成原始特征
        self.features = torch.randn(num_samples, feature_dim)
        
        # 2. 抽取子特征并计算 score (每 100 维取一个)
        # TODO: 这里写你的切片和 mean 逻辑
        self.scores = self.features[:, ::100].mean(dim=1)
        
        # 3. 计算分位数并生成标签
        # TODO: 这里写 quantile 和 标签分配逻辑
        # 确保 self.labels 的 dtype 是 torch.long
        quantiles = torch.quantile(self.scores, torch.tensor([0.33, 0.66])) # 计算 33% 和 66% 的分位数, quantities 是一个包含两个元素的 tensor，分别是 33% 和 66% 的分位数
        self.labels = torch.zeros(num_samples, dtype=torch.long) #torch.zeros 创建一个全零的 tensor，大小是 num_samples，dtype=torch.long 表示这个 tensor 的数据类型是 long（整数类型）
        self.labels[self.scores > quantiles[0]] = 1
        self.labels[self.scores > quantiles[1]] = 2
        pass

    def __len__(self):
        """
        第二板斧：告诉 DataLoader 这个数据集一共有多少个样本。
        """
        # TODO: 返回总样本数 (num_samples)
        return len(self.features)

    def __getitem__(self, idx):
        """
        第三板斧：核心！DataLoader 每次来要数据，就会传一个索引 idx 进来。
        你必须根据这个 idx，返回对应的“一个”特征和“一个”标签。
        """
        # TODO: 根据 idx 提取出对应的特征 (x) 和标签 (y)
        x = self.features[idx]
        y = self.labels[idx]
        return x, y
