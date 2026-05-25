from torch import nn
#===============一个简单的 MLP 分类器，作为 router 的第一个 baseline======================

class SimpleRouter(nn.Module):
    """
    一个很小的 MLP 分类器，作为当前的第一个 router baseline。

    网络结构:
    input -> Linear -> ReLU -> Dropout -> Linear -> logits
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int): #网络层定义
        """
        Args:
            input_dim: 输入特征维度。
            hidden_dim: 隐藏层宽度。
            num_classes: 输出类别数，也可以理解成路由选择数。
        """
        super().__init__() # 这行是必须的，调用父类的构造函数，确保 nn.Module 的内部机制正常工作。

        # nn.Sequential 会按顺序把下面这些层串起来。
        self.network = nn.Sequential(
            # 第一层线性层:
            # 把原始输入从 input_dim 映射到 hidden_dim。
            nn.Linear(input_dim, hidden_dim),

            # ReLU 非线性激活:
            # 让模型不只是简单线性变换，表达能力更强。
            nn.ReLU(),

            # 加入 dropout，随机丢弃一些神经元，帮助防止过拟合
            nn.Dropout(p=0.1),  # 10% 的概率丢弃神经元，剩下的神经元输出会被放大到原来的 1/(1-0.1)=1.11 倍，以保持整体输出的期望不变。
            
            # 第二层线性层:
            # 把隐藏表示映射成最终的类别分数 logits。
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs): # forward 方法定义了模型的前向传播逻辑
        """
        Args:
            inputs: 输入张量，形状是 [batch_size, input_dim]。

        Returns:
            输出 logits，形状是 [batch_size, num_classes]。
        """
        return self.network(inputs)
