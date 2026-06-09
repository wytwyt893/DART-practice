"""
Text Expert prompt 构造工具。

这个文件服务于 Stage 5.3A：先复现官方 TableDART 的 Text Expert 输入格式，
再考虑真正提取 text_gate_feats。

为什么要单独写这个文件？
1. 官方 TableDART 不是直接把 question 送进 Text Expert。
2. 它会先把 table 转成 Markdown 文本。
3. 再根据不同 benchmark category 选择不同 prompt 模板。
4. 最后得到 prompt_for_text_expert，送给 TableGPT2Expert。

官方链路可以简化成：

    raw jsonl sample
    -> table / question / category
    -> markdown table
    -> category-specific prompt template
    -> prompt_for_text_expert
    -> TableGPT2Expert.extract_gate_features_and_intermediate_state(...)
    -> text_gate_feats

我们当前阶段先只复现到 prompt_for_text_expert。
下一阶段才会做：

    prompt_for_text_expert
    -> tokenizer / text model
    -> text_features.pt

注意：
这个文件里的 prompt 模板来自官方 TableDART 的
TableDART-Official/TableDART/models/prompt/markdown_prompt.py。
为了让主项目不依赖官方仓库的 Python import 路径，我们在这里保留了一份本地映射。
"""

from __future__ import annotations

from typing import Any


# -----------------------------
# 1. Prompt templates
# -----------------------------
#
# 不同 benchmark 的任务形式不一样：
# - WTQ / HiTab / TAT-QA: 表格问答
# - TabFact / InfoTabs: 表格事实验证
# - TABMWP: 表格数学题
# - FeTaQA: 生成式表格问答
#
# 所以官方会给不同 category 使用不同的 prompt 模板。
# 每个模板里通常会有两个占位符：
# - {md_table}: Markdown 格式的表格文本
# - {question}: 原始问题 / statement / hypothesis

WTQ_TEMPLATE = """
You are an expert table question-answering assistant. Your task is to analyze the provided Markdown table and the user's question, then generate a structured JSON response.

**Instructions:**
1. Analyze the Question: Carefully understand what information the user is asking for from the table.
2. Extract the Answer: Locate the precise information in the table that directly answers the question.
3. Formulate the Explanation: Briefly describe, in a single sentence, the steps you took to find the answer in the table.
4. Construct the JSON Output: Create a single, valid JSON object with keys "answer" and "explanation".

**Important:** Your final output must only be the JSON object.

**Table:**
```
{md_table}
```

**Question:**
{question}

**JSON Answer:**
""".lstrip()

TABFACT_TEMPLATE = """
You are an expert table fact verification assistant. Your task is to analyze the provided Markdown table and the given statement, then determine if the statement is factually supported by the table.

**Instructions:**
1. Analyze the Statement.
2. Examine the Table.
3. Decide whether the statement is True or False.
4. Return a JSON object with keys "answer" and "explanation".

**Important:** Your final output must only be the JSON object.

**Table:**
```
{md_table}
```

**Statement:**
{question}

**JSON Answer:**
""".lstrip()

FETAQA_TEMPLATE = """
Markdown Table:
{md_table}

Question: {question}
Provide a direct, concise answer:
""".lstrip()

TABMWP_TEMPLATE = """
You are an expert table question-answering assistant specializing in mathematical reasoning. Your task is to analyze the provided Markdown table and the user's math problem, then generate a structured JSON response.

**Instructions:**
1. Analyze the Math Problem.
2. Perform the Calculation using the table data.
3. Extract the numerical or text answer.
4. Return a JSON object with keys "answer" and "explanation".

**Important:** Your final output must only be the JSON object.

**Table:**
```
{md_table}
```

**Math Problem:**
{question}

**JSON Answer:**
""".lstrip()

HITAB_TEMPLATE = """
You are an expert table question-answering assistant specializing in hierarchical tables. Analyze the complex Markdown table and answer the user's question.

**Table Structure Notes:**
- The table may have hierarchical headers.
- Some cells may be merged or span multiple rows/columns.
- Pay attention to the table structure.

**Table:**
```
{md_table}
```

**Question:**
{question}

**JSON Answer:**
""".lstrip()

INFOTABS_TEMPLATE = """
You are an expert table fact verification assistant. Your task is to analyze the provided Markdown table and the given hypothesis, then determine the logical relationship between the hypothesis and the table.

The answer should be one of: Entail, Contradict, or Neutral.

**Table:**
```
{md_table}
```

**Hypothesis:**
{question}

**JSON Answer:**
""".lstrip()

TATQA_TEMPLATE = """
You are an expert table question-answering assistant specializing in mathematical reasoning. Analyze the Markdown table and answer the user's question.

**Table:**
```
{md_table}
```

**Question:**
{question}

**JSON Answer:**
""".lstrip()

DEFAULT_MARKDOWN_PROMPT_BODY = """
Markdown Table:
{md_table}

Given the table above and the question: {question}
Provide only the direct answer.
""".lstrip()


MARKDOWN_TEMPLATE_MAPPING = {
    "WTQ_for_TQA": WTQ_TEMPLATE,
    "TABMWP_for_TQA": TABMWP_TEMPLATE,
    "TAT-QA_for_TQA": TATQA_TEMPLATE,
    "HiTab_for_TQA": HITAB_TEMPLATE,
    "FeTaQA_for_TQA": FETAQA_TEMPLATE,
    "InfoTabs_for_TFV": INFOTABS_TEMPLATE,
    "TabFact_for_TFV": TABFACT_TEMPLATE,
}


def _stringify_cell(value: Any) -> str:
    """
    把表格单元格转换成 Markdown 里安全显示的字符串。

    为什么需要这个函数？
    原始 table 里的 cell 可能不是普通字符串，可能是数字、None、list、dict。
    Markdown 表格最后只能拼接字符串，所以这里统一转成 str。

    另外，Markdown 表格用竖线 `|` 分隔列。
    如果 cell 内容本身包含 `|`，会破坏表格结构，所以这里替换成 `/`。
    """
    if value is None:
        return ""
    return (
        str(value)
        .replace("\xa0", " ")
        .replace("|", "/")
        .replace("\n", " ")
        .strip()
    )


def _truncate_text(text: str, token_limit: int) -> str:
    """
    对 Markdown 文本做长度截断。

    官方代码用的是一个经验近似：
        如果 markdown 字符长度 > token_limit * 5，就截断。

    这里继续沿用这个思想。
    它不是严格 tokenizer 级别的 token 截断，但足够作为 prompt 构造阶段的安全保护，
    避免特别长的表格把 prompt 撑爆。
    """
    char_limit = token_limit * 5
    if len(text) > char_limit:
        return text[:char_limit] + "\n...[truncated]"
    return text


def table_to_markdown(table: dict, token_limit: int = 3800) -> str:
    """
    把普通表格转换成 Markdown table。

    输入格式来自官方 jsonl 的 table 字段，通常长这样：

        {
            "header": ["Name", "Country", "Rank"],
            "rows": [
                ["Alice", "Italy", "1"],
                ["Bob", "France", "2"]
            ]
        }

    输出格式类似：

        | Name | Country | Rank |
        | --- | --- | --- |
        | Alice | Italy | 1 |
        | Bob | France | 2 |

    Text Expert 看到的不是 Python dict，而是这种 Markdown 字符串。
    """
    if not isinstance(table, dict):
        return ""

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not isinstance(header, list) or len(header) == 0:
        return ""
    if not isinstance(rows, list):
        rows = []

    clean_header = [_stringify_cell(cell) for cell in header]
    num_cols = len(clean_header)

    markdown_lines = [
        "| " + " | ".join(clean_header) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]

    for row in rows:
        if not isinstance(row, list):
            continue

        # 如果某行列数和 header 不一致，官方 pandas 版本会更宽容地处理。
        # 这里为了让 Markdown 结构稳定，做一个简单对齐：
        # - 列太少：补空字符串
        # - 列太多：截断到 header 长度
        aligned_row = row[:num_cols] + [""] * max(0, num_cols - len(row))
        markdown_lines.append(
            "| " + " | ".join(_stringify_cell(cell) for cell in aligned_row) + " |"
        )

    return _truncate_text("\n".join(markdown_lines), token_limit=token_limit)


def infotabs_table_to_markdown(table: dict, token_limit: int = 3800) -> str:
    """
    InfoTabs 的 table 格式和普通 row/column 表格不完全一样。

    官方代码会把它转成两列表：
        | Attribute | Value |
        | --- | --- |
        | key1 | value1 |
        | key2 | value2 |

    这样 Text Expert 可以用统一的 Markdown 方式读取属性型表格。
    """
    if not isinstance(table, dict):
        return ""

    markdown_lines = [
        "| Attribute | Value |",
        "| --- | --- |",
    ]

    for key, value in table.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, list):
            value_text = " ".join(_stringify_cell(v) for v in value)
        else:
            value_text = _stringify_cell(value)
        markdown_lines.append(f"| {_stringify_cell(key)} | {value_text} |")

    return _truncate_text("\n".join(markdown_lines), token_limit=token_limit)


def build_text_expert_prompt(
    table: dict,
    question: str,
    category: str,
    token_limit: int = 3800,
) -> str:
    """
    构造官方 Text Expert 使用的 prompt_for_text_expert。

    这是 Stage 5.3A 的核心函数。

    数据流：
        table + question + category
        -> Markdown table
        -> category-specific prompt template
        -> prompt_for_text_expert

    参数说明：
    - table:
      原始 jsonl 样本里的 table 字段。
    - question:
      原始问题 / statement / hypothesis。
    - category:
      数据集类别，例如 WTQ_for_TQA、TabFact_for_TFV。
      不同类别会使用不同 prompt 模板。
    - token_limit:
      Markdown 表格的近似长度保护参数。

    返回：
    - 一个字符串 prompt。
      后续 Text Expert 会对这个字符串做 tokenizer / embedding / pooling。
    """
    if "InfoTabs" in category:
        markdown_table = infotabs_table_to_markdown(table, token_limit=token_limit)
    else:
        markdown_table = table_to_markdown(table, token_limit=token_limit)

    template = MARKDOWN_TEMPLATE_MAPPING.get(category, DEFAULT_MARKDOWN_PROMPT_BODY)

    try:
        return template.format(md_table=markdown_table, question=question)
    except KeyError:
        # 如果某个模板占位符写得不兼容，就退回最简单、最稳的格式。
        return DEFAULT_MARKDOWN_PROMPT_BODY.format(
            md_table=markdown_table,
            question=question,
        )
