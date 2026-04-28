# 导入需要的模块
import torch
from torch.utils.data import DataLoader

# 从 data.dataloader 导入 SyntheticDataset
from data.dataloader import SyntheticDataset

def main():
    # 1. 创建数据集
    data = SyntheticDataset(num_samples=1000, feature_dim=10112, num_classes=3)

    # 2. 创建 DataLoader
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    # 3. 取一个 batch
    batch = next(iter(dataloader)) #iter(dataloader) 创建一个迭代器，next() 从中取出一个 batch
    features, labels = batch # batch 是一个 tuple，包含 features 和 labels 两部分，我们把它们分别赋值给 features 和 labels 变量

    # 4. 打印 shape 和 dtype
    print("Features shape:", features.shape) # 应该是 (32, 10112)
    print("Features dtype:", features.dtype) # 应该是 torch.float32
    print("Labels shape:", labels.shape) # 应该是 (32,)
    print("Labels dtype:", labels.dtype) # 应该是 torch.int64
   
   # 5. 提前结束，只检查第一批
    print("第一批数据检查完成。")

if __name__ == "__main__":
    main()
