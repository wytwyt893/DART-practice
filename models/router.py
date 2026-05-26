from torch import nn
#===============一个简单的 MLP 分类器，作为 router 的第一个 baseline======================

class MLPRouter(nn.Module):
    """
    一个很小的 MLP 分类器，作为当前的第一个 router baseline。

    网络结构:
    input -> Linear -> ReLU -> Dropout -> Linear -> logits
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_routes: int, dropout: float): #网络层定义
        """
        Args:
            input_dim: 输入特征维度。
            hidden_dim: 隐藏层宽度。
            num_routes: 输出路由数，也可以理解成路由选择数。
            dropout: Dropout 概率。
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
            nn.Dropout(p=dropout),
            
            # 第二层线性层:
            # 把隐藏表示映射成最终的类别分数 logits。
            nn.Linear(hidden_dim, num_routes),
        )

    def forward(self, inputs): # forward 方法定义了模型的前向传播逻辑
        """
        Args:
            inputs: 输入张量，形状是 [batch_size, input_dim]。

        Returns:
            输出 logits，形状是 [batch_size, num_routes]。
        """
        return self.network(inputs)
