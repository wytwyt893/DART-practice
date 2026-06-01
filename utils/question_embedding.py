import torch
from transformers import AutoModel, AutoTokenizer


_TOKENIZER = None
_MODEL = None
_MODEL_ID = None


def get_question_embedding_for_gate(
    question_text: str,
    device: torch.device,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> torch.Tensor:
    """
    使用 all-MiniLM-L6-v2 把 question 文本编码成 gate 使用的 384 维向量。

    流程对齐官方 TableDART:
    question text
    -> tokenizer
    -> AutoModel
    -> last_hidden_state
    -> attention-mask mean pooling
    -> [384] question embedding

    数据流格式变化为：
    question_text: str
    -> inputs: dict of tensors (input_ids, attention_mask, etc.)
        -> token_embeddings: torch.Tensor of shape [1, seq_len, 384]
            -> attention_mask: torch.Tensor of shape [1, seq_len, 1]
            -> pooled: torch.Tensor of shape [1, 384]
    """
    global _TOKENIZER, _MODEL, _MODEL_ID

    # 这里做了一个简单的缓存：
    # 如果连续调用这个函数，且 model_id 没变，就不会重复加载模型和 tokenizer。
    if _TOKENIZER is None or _MODEL is None or _MODEL_ID != model_id:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _MODEL = AutoModel.from_pretrained(model_id).to(device)
        _MODEL.eval()
        _MODEL_ID = model_id

    # 把 question text 编码成模型输入格式，移动到 device 上。
    inputs = _TOKENIZER(
        question_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(device)

    # 通过模型前向传播得到 token embeddings
    # 然后做 attention mask 加权平均池化，得到最终的 question embedding。
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    
    # outputs.last_hidden_state 的 shape 是 [1, seq_len, hidden_dim]
    # 其中 hidden_dim=384 是 all-MiniLM-L6-v2 的输出维度。
    token_embeddings = outputs.last_hidden_state

    # mask处理：attention_mask 的 shape 是 [1, seq_len]，值为 1 的位置是有效 token，值为 0 的位置是 padding。
    attention_mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)

    # 加权平均池化的目的是：把 seq_len 维度的 token embedding 聚合成一个固定维度的 question embedding。
    pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

    return pooled.squeeze(0).detach().cpu()