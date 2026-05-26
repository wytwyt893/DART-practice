import random
import torch

def set_seed(seed: int = 42) -> None:
    """
    固定随机种子，尽量让每次运行结果保持一致，方便复现实验。

    Args:
        seed: 同时作用于 Python random 和 PyTorch 的随机种子。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)