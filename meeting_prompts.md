# Group Meeting Diagram Prompts

## 1. Training Pipeline Flowchart Prompt

Use this prompt in a diagram tool or image generator:

```text
Draw a clean academic flowchart for a machine learning sanity-check pipeline on a white background.
Style: neat research-slide figure, minimal color palette, blue and orange accents, thin arrows, rounded rectangles, legible English labels.

Pipeline from left to right:
1. SyntheticDataset
   subtitle: random features + class-dependent signal injection
2. DataLoader
   subtitle: batch_size=64, shuffle for train
3. SimpleRouter MLP
   subtitle: Linear(1024 -> 256) + ReLU + Linear(256 -> 3)
4. CrossEntropyLoss
   subtitle: compare logits with labels
5. Backpropagation + Adam
   subtitle: lr=1e-3
6. Metrics Output
   subtitle: train_loss, train_acc, val_loss, val_acc

Add a small note under the whole figure:
"Goal: verify that the minimal training pipeline is runnable before moving to real data and dynamic routing."

Make it look like a polished conference slide diagram, not a hand-drawn sketch.
```

## 2. Current Network Structure Prompt

Use this prompt if you want a separate architecture figure:

```text
Draw a simple neural network architecture diagram for a baseline MLP classifier.
White background, academic presentation style, clean layout, blue and gray palette.

Show the model from left to right:
Input feature vector (1024-d)
-> Linear layer (1024 to 256)
-> ReLU activation
-> Linear layer (256 to 3)
-> Output logits for 3 classes

Annotate clearly:
- input_dim = 1024
- hidden_dim = 256
- num_classes = 3
- this is a baseline MLP, not yet a dynamic routing model

Add a footer note:
"Used only for synthetic sanity check."

Keep the figure minimal, professional, and easy to place on a research group meeting slide.
```

## 3. One-Line Chinese Explanation for Slides

- 流程图配文: 当前已完成 synthetic 数据到训练验证输出的最小实验闭环。
- 网络图配文: 当前模型是最小 MLP baseline，用于验证训练链路，不是最终的双层动态路由结构。
