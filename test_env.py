import torch
import torchvision
import numpy
import wandb
import tqdm

#===================================环境检查====================================
# 1. 检查各库的版本
print("PyTorch 版本:", torch.__version__)
print("Torchvision 版本:", torchvision.__version__)
print("NumPy 版本:", numpy.__version__)
print("WandB 版本:", wandb.__version__)
print("tqdm 版本:", tqdm.__version__)

# 2. 检查显卡 (GPU/CUDA) 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

# 如果 CUDA 可用，打印显卡信息
if torch.cuda.is_available():
    print("可用显卡数量:", torch.cuda.device_count())
    print("当前显卡型号:", torch.cuda.get_device_name(0))
