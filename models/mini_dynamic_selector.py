import torch
from torch import nn

from models.router import MLPRouter


class MiniDynamicSelector(nn.Module):#nn.Module 是 PyTorch 中所有神经网络模块的基类，我们定义的 MiniDynamicSelector 继承自 nn.Module，意味着它是一个可训练的神经网络模型，可以包含参数、子模块，并且可以在训练过程中更新这些参数。
    """
    Mini version of TableDART DynamicExpertSelector.

    当前版本不接真实专家模型，只模拟官方 dynamic_selector.py 里的核心数据流：
        text_feature + image_feature + question_feature
        -> concat
        -> MLPRouter
        -> route logits
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        question_dim: int,
        hidden_dim: int,
        num_routes: int,
        dropout: float,
    ):
        super().__init__()

        # 三路 feature 在 forward 里会沿 dim=1 拼接，
        # 所以 router 的输入维度等于三路特征维度之和。
        input_dim = text_dim + image_dim + question_dim

        # self.router 是从 models/router.py 导入的 MLPRouter 子模块。
        #
        # 这里的 input_dim/hidden_dim/num_routes/dropout 是“建网络结构”用的参数，
        # 会在 MLPRouter.__init__ 里创建 Linear/ReLU/Dropout/Linear。
        #
        # 后面 forward 里写 self.router(combined_feature) 时，
        # 实际会跨文件调用 models/router.py 里的 MLPRouter.forward(combined_feature)。
        self.router = MLPRouter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_routes=num_routes,
            dropout=dropout,
        )

    def forward(
        self,
        text_feature: torch.Tensor,
        image_feature: torch.Tensor,
        question_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feature: [batch_size, text_dim]
            image_feature: [batch_size, image_dim]
            question_feature: [batch_size, question_dim]

        Returns:
            route_logits: [batch_size, num_routes]
        """
        # 在当前文件里先完成三路特征拼接。
        # dim=1 表示沿“特征维度”拼接，而不是沿 batch 维度拼接。
        combined_feature = torch.cat(
            [text_feature, image_feature, question_feature],
            dim=1,
        )

        # 这里进入跨文件调用：
        # self.router 是 MLPRouter 对象，所以这一行会调用 models/router.py 里的
        # MLPRouter.forward(inputs)，并返回 [batch_size, num_routes] 的 logits。
        route_logits = self.router(combined_feature)
        return route_logits
