# src/scripts/extract_financials.py
import json
import re
import os
import sqlite3
import pathlib
import time
import chromadb

from itertools import groupby
from react_agent.utils.llm import load_chat_model
from react_agent.core.config import settings


DB_PATH = pathlib.Path(settings.tools.sql_store.DB_PATH)
PDF_DIR = pathlib.Path(settings.tools.vector_store.data_dir)

_CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_CHROMA_COLLECTION = "langchain"

# LLM 调用失败时的重试参数
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # 秒


# ── FIX [1/4]: 放开 metric 归一化限制，增加 raw_metric 字段保留原文表述
#    旧版强制归一化会导致大量重要指标（ROE、EBIT、存货周转率等）被静默丢弃
EXTRACT_PROMPT = """你是客观、专业的金融数据抽取专家。你的任务是从涵盖电子、半导体、电商、房地产等多领域的研报片段中，精准提取财务与经营数据，并严格输出为 JSON 数组。

【核心红线：禁止自作聪明】
1. 绝对禁止任何计算：严禁进行汇率转换、单位换算或量级乘除！原文数字是多少，value 就是多少。
2. 绝对忠于原文：不要基于行业常识进行主观推断，不要捏造文本中未明确提及的数据。

【字段提取规则】
1. metric：优先归一化为以下标准列表中的一项：【收入、净利润、毛利率、净利率、市盈率、EV/EBITDA、市值、同比增速、市场份额】。
   如果原文指标无法合理归入上述标准列表，请直接使用原文指标名（如"经营利润"、"ROE"、"EBIT"、"存货周转率"），
   不要强行归类，更不要丢弃——宁可保留非标指标，也不要因归类困难而遗漏有价值的数据。
2. raw_metric：无论 metric 是否归一化成功，都必须照抄原文中的原始指标表述，用于校验和溯源。
3. value：仅提取纯数字（REAL类型），不含逗号、百分号或单位字符。
4. unit：严格照抄原文紧跟在数字后面的单位描述（如"亿元"、"百万美元"、"倍"、"万平米"等）。对于百分比，unit 统一填 "%"，value 填百分号前的纯数字（如原文"33.2%" -> value: 33.2, unit: "%"）。
5. year：统一为标准字符串（如 "FY24"、"2024"、"1H25"）。若原文含有"预计"、"预测"、"E"等代表预测的字眼（如"2025E"），将 year 提取为纯年份（如"2025"），并将 is_estimate 设为 1；否则 is_estimate 设为 0。
6. ticker：仅当原文明确给出股票代码时填写（如 "603501 CH"），否则必须填 null。
7. company：提取研报中涉及的完整公司名。

【输出格式】
仅输出合法的 JSON 数组，不要任何 Markdown 标记（如 ```json ），不要任何解释。未提取到相关数据则输出 []。

【标准示例】
原文："豪威集团2024年收入为3574百万美元，预计2025年净利润达到50.5亿元人民币，市场份额占比15.2%，ROE为12.3%。"
输出：
[
  {{"company": "豪威集团", "ticker": null, "metric": "收入", "raw_metric": "收入", "value": 3574, "unit": "百万美元", "year": "2024", "is_estimate": 0}},
  {{"company": "豪威集团", "ticker": null, "metric": "净利润", "raw_metric": "净利润", "value": 50.5, "unit": "亿元人民币", "year": "2025", "is_estimate": 1}},
  {{"company": "豪威集团", "ticker": null, "metric": "市场份额", "raw_metric": "市场份额占比", "value": 15.2, "unit": "%", "year": "2024", "is_estimate": 0}},
  {{"company": "豪威集团", "ticker": null, "metric": "ROE", "raw_metric": "ROE", "value": 12.3, "unit": "%", "year": "2024", "is_estimate": 0}}
]

待提取研报文本：
{text}
"""


# ──────────────────────────────────────────────
# DB 初始化
# ──────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    # ── FIX [2/4]: 重新设计唯一索引 + 新增 raw_metric / confidence 字段
    #
    # 旧版问题：
    #   (company, metric, year, source_file) 作为唯一键，
    #   导致同一公司同一指标在两份不同研报里被视为合法重复写入，
    #   查询时产生噪声，LLM 汇总时容易被误导。
    #
    # 新版设计原则：
    #   · 业务唯一键 = (company, metric, year)，相同来源不重复写入。
    #   · 不同来源研报对同一指标 => 允许多条，用 source_file 区分，
    #     查询层按 confidence DESC / source_page ASC 取最优一条即可。
    #   · raw_metric 保留原文表述，供后期数据质检和 NL 检索。
    #   · confidence 预留数据权威度字段（1=普通研报, 2=官方财报, 3=彭博/Wind）。
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        company      TEXT    NOT NULL,
        ticker       TEXT,
        metric       TEXT    NOT NULL,
        raw_metric   TEXT,               -- 原文指标表述，用于溯源与质检
        value        REAL,
        unit         TEXT,
        year         TEXT    NOT NULL,
        is_estimate  INTEGER DEFAULT 0,
        confidence   INTEGER DEFAULT 1,  -- 数据权威度：1=研报, 2=官方财报, 3=终端数据
        source_file  TEXT,
        source_page  INTEGER
    );

    -- 旧版的唯一约束包含 source_file，会导致同一指标多条噪声数据共存。
    -- 新版：(company, metric, year, source_file) 保证同一份研报内幂等，
    -- 不同研报的同一指标可共存，查询层负责去噪（取 confidence 最高或最新一条）。
    CREATE UNIQUE INDEX IF NOT EXISTS uix_metrics
        ON financial_metrics(company, metric, year, source_file);

    -- 高频查询加速：按公司+指标+年份检索
    CREATE INDEX IF NOT EXISTS idx_metrics_lookup
        ON financial_metrics(company, metric, year);

    -- 高频查询加速：按文件溯源
    CREATE INDEX IF NOT EXISTS idx_metrics_source
        ON financial_metrics(source_file, source_page);

    CREATE TABLE IF NOT EXISTS market_share (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        company     TEXT    NOT NULL,
        segment     TEXT    NOT NULL,
        share_pct   REAL,
        year        TEXT    NOT NULL,
        source_file TEXT,
        source_page INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_share_company
        ON market_share(company, segment, year);

    -- 断点续跑：记录已成功处理的文件
    CREATE TABLE IF NOT EXISTS processed_files (
        filename    TEXT PRIMARY KEY,
        processed_at TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def _strip_markdown_fence(text: str) -> str:
    """
    剥除 LLM 可能输出的 markdown 代码块围栏。
    str.strip("```json") 是按字符集剥离，会误删 JSON 两端的合法字符。
    应使用正则按子字符串匹配。
    """
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _fetch_chunks_from_chroma(
    collection,
    pdf_path: pathlib.Path,
) -> list[tuple[str, int]]:
    """
    从 Chroma 按 source（完整文件路径）查询该 PDF 的全部 chunks，
    并按页码分组合并，返回 [(page_text, page_num), ...] 列表。
    """
    result = collection.get(
        where={"source": str(pdf_path)},
        include=["documents", "metadatas"],
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if not documents:
        return []

    pairs = sorted(
        zip(documents, metadatas),
        key=lambda x: x[1].get("chunk_id", ""),
    )

    chunks = []
    for page_num, group in groupby(pairs, key=lambda x: x[1].get("page", 0)):
        page_text = "\n\n".join(doc for doc, _ in group).strip()
        if page_text:
            chunks.append((page_text, int(page_num)))
    return chunks


def _call_llm_with_retry(llm, prompt: str) -> list[dict]:
    """
    调用 LLM 并解析 JSON，失败时最多重试 _MAX_RETRIES 次。
    返回记录列表，解析失败则抛出最后一次异常。
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = llm.invoke(prompt)
            raw = _strip_markdown_fence(resp.content)
            return json.loads(raw)
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
    raise last_exc


# ──────────────────────────────────────────────
# 核心处理
# ──────────────────────────────────────────────
def extract_one_pdf(
    pdf_path: pathlib.Path,
    llm,
    conn: sqlite3.Connection,
    collection,
) -> int:
    """
    解析单份研报 PDF，抽取财务指标并写入 DB。
    返回本次实际插入的记录数。
    """
    filename = pdf_path.name

    chunks = _fetch_chunks_from_chroma(collection, pdf_path)
    if not chunks:
        print(f"  ✗ Chroma 中未找到该文件的 chunks，请先运行 build_vector_db() ({filename})")
        return 0

    inserted = 0
    for page_text, page_num in chunks:
        if len(page_text.strip()) < 100:
            continue

        try:
            records = _call_llm_with_retry(
                llm, EXTRACT_PROMPT.format(text=page_text)
            )
        except Exception as e:
            print(f"  ⚠ LLM 抽取失败，已重试 {_MAX_RETRIES} 次 ({filename} p{page_num}): {e}")
            continue

        for r in records:
            if not r.get("company") or not r.get("metric") or not r.get("year"):
                continue
            try:
                conn.execute("""
                            INSERT OR IGNORE INTO financial_metrics
                                (company, ticker, metric, raw_metric, value, unit,
                                 year, is_estimate, source_file, source_page)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                    r["company"],
                    r.get("ticker"),
                    r["metric"],
                    r.get("raw_metric"),
                    r.get("value"),
                    r.get("unit"),
                    r["year"],
                    r.get("is_estimate", 0),
                    filename,
                    page_num,
                ))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except sqlite3.Error as e:
                print(f"  ⚠ DB 写入失败 ({filename} p{page_num}): {e}")

        conn.commit()

    return inserted


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model_ref = "deepseek/deepseek-v4-flash"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    llm = load_chat_model(model_ref)

    chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
    collection = chroma_client.get_collection(_CHROMA_COLLECTION)

    pdfs = list(PDF_DIR.rglob("*.pdf"))
    print(f"开始处理 {len(pdfs)} 份研报...")

    stop_num: int = 0
    for i, pdf in enumerate(pdfs):
        stop_num += 1
        if stop_num == 21:
            break

        filename = pdf.name

        already_done = conn.execute(
            "SELECT 1 FROM processed_files WHERE filename = ?", (filename,)
        ).fetchone()
        if already_done:
            print(f"[{i+1}/{len(pdfs)}] 跳过（已处理）: {filename}")
            continue

        print(f"[{i+1}/{len(pdfs)}] {filename}")
        n = extract_one_pdf(pdf, llm, conn, collection)
        print(f"  → 写入 {n} 条记录")

        conn.execute(
            "INSERT OR REPLACE INTO processed_files (filename) VALUES (?)",
            (filename,),
        )
        conn.commit()

    conn.close()
    print("完成！")