# eval/ — RAG 评测模块

# 本模块为 ReAct Agent 项目提供标准化的 RAG 评测能力，涵盖四项核心 RAGAS 指标。

## 目录结构

'''
eval/
├── __init__.py
├── run_eval.py           # 统一入口（推荐从这里运行）
├── dataset_generator.py  # 阶段一：合成 QA 数据集生成
├── evaluator.py          # 阶段二：RAGAS 评测执行
└── dataset/              # 数据集存放目录（自动创建）
    └── eval_dataset.jsonl

results/                  # 评测结果（自动创建，在 src/eval/results/）
├── eval_summary_YYYYMMDD_HHMMSS.json
└── eval_detail_YYYYMMDD_HHMMSS.csv
'''

## 安装额外依赖

'''bash
pip install ragas datasets langchain-huggingface
'''

## 快速开始

### 第一步：生成评测数据集（一次性）

'''bash
cd src
python eval/run_eval.py generate \
    --data_dir   data \
    --output     eval/dataset/eval_dataset.jsonl \
    --max_chunks 150 \
    --n_per_chunk 2


耗时约 30~60 分钟（取决于 API 速度和 chunk 数量）。  
生成后数据集固定，可反复用于不同参数配置的评测对比。

### 第二步A：仅评检索质量（快，推荐日常迭代使用）

```bash
python eval/run_eval.py eval \
    --retrieval_only \
    --n_samples 50
```

只计算 Context Precision 和 Context Recall，不调用 LLM 生成答案。  
适合调优 `RERANKER_THRESHOLD`、`chunk_size`、BM25 权重等检索参数时快速反馈。

### 第二步B：端到端评测（全面，含 LLM 生成）

```bash
python eval/run_eval.py eval \
    --n_samples 50
```

同时计算全部四项指标，包括 Faithfulness 和 Answer Relevancy。

### 查看最近报告

```bash
python eval/run_eval.py report
```

## 四项指标说明

| 指标 | 含义 | 低分对应问题 | 调优方向 |
|------|------|-------------|---------|
| Context Precision | 检索到的 chunks 有多少是真正相关的 | 检索噪声多 | 提高 RERANKER_THRESHOLD |
| Context Recall | ground_truth 所需信息有多少被检索到 | 漏召 | 增大 top_k，减小 chunk_size |
| Faithfulness | 答案是否完全由 context 支撑 | 模型幻觉 | 强化 Prompt 拒答指令 |
| Answer Relevancy | 答案是否真正回答了问题 | 答非所问 | 检查 system prompt 输出约束 |

## 评测数据集格式

每行一条 JSONL，字段如下：

```json
{
  "question":     "比亚迪2023年Q3毛利率是多少？",
  "ground_truth": "22.1%，同比提升3.2个百分点。",
  "source_file":  "半导体/中芯国际_2023Q3点评.pdf",
  "page":         5,
  "industry":     "半导体",
  "chunk_text":   "...(原始chunk内容)..."
}
'''