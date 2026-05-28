# TableDART Baseline Notes

## 1. 论文基本信息

- 论文名称：
- 论文链接：
- 代码仓库：
- 任务类型：
- 数据集：
- 主要评价指标：

## 2. TableDART 的核心问题

用自己的话说明这篇论文解决什么问题。

例如：

- 输入是什么？
- 输出是什么？
- 为什么普通 MLLM / VLM 对表格推理不够？
- TableDART 想通过什么机制改进？

## 3. 总体架构拆解

把 TableDART 拆成几个模块：

- 输入处理 / 特征构造
- Router / Dynamic Routing
- Text route
- Image route
- Fusion route
- Decoder / Answer prediction
- Loss / Training objective

每个模块下面写：

- 输入：
- 输出：
- 作用：
- 复现难度：

## 4. Router Module 重点整理

这是和你当前 mini 项目最相关的部分。

需要重点记录：

- Router 的输入维度：
- Router 输入由哪些特征组成：
- Router 网络结构：
- Router 输出类别：
- 路由标签如何构造：
- Router loss：
- 是否和主任务联合训练：
- 是否有额外正则项：

## 5. 和当前 mini 项目的对应关系

| TableDART 组件 | 当前项目对应实现 | 当前状态 | 缺口 |
|---|---|---|---|
| Router MLP | `models/router.py` | 已有 mini 版 | 需要对齐真实输入 |
| Router 输入特征 | `SyntheticDataset.features` | synthetic | 需要真实特征 |
| Router 标签 | `route_labels` | synthetic | 需要真实监督或规则 |
| Train loop | `train.py` | 已有 | 需要接真实数据 |
| Eval loop | `eval.py` | 已有 | 需要真实指标 |

## 6. 真实复现需要的数据和资源

- 需要哪些数据集？
- 数据格式是什么？
- 是否需要图像表格？
- 是否需要 OCR？
- 是否需要预提取特征？
- 训练需要多大显存？
- 是否需要租卡？

## 7. 第一阶段可复现目标

先不要写“完整复现 TableDART”。  
建议写成：

- 目标 A：复现 Router MLP 结构
- 目标 B：用真实/半真实特征替换 synthetic features
- 目标 C：复现 route prediction accuracy
- 目标 D：再考虑接下游任务指标

## 8. Open Questions

把你现在还不确定的问题列出来：

- 10112 维具体由哪些部分拼接？
- route label 是人工标注、规则生成还是从性能选择得到？
- 三路 decoder 是否需要同时训练？
- 官方代码是否公开？
- 数据预处理成本多大？