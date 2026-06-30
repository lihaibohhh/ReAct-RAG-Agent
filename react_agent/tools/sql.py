# src/react_agent/tools/sql.py
import json
import sqlite3
import re
import os
from langchain_core.tools import tool
from react_agent.utils.tool_helpers import _ok, _err
from react_agent.core.config import settings


DB_PATH = settings.tools.sql_store.DB_PATH

# 安全白名单：只允许 SELECT
_SAFE_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

_TOOL_NAME = "sql_tool"

model_ref = os.getenv("MODEL", "deepseek/deepseek-v4-flash")

def _extract_sql(raw: str) -> str:
    """
    从 LLM 输出中提取 SQL，兼容以下格式：
    1. 纯 SQL（无任何包裹）
    2. ```sql ... ```
    3. ``` ... ```
    4. SQL 前后带有说明文字
    """
    # 优先匹配代码块（含或不含语言标注）
    code_block = re.search(r"```(?:sql)?\s*\n?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    # 没有代码块时，找第一个 SELECT 语句到结尾
    select_match = re.search(r"(SELECT\b.*)", raw, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    # 兜底：直接返回原始内容并交给安全校验去拦截
    return raw.strip()


def _query_db(sql: str) -> list[dict]:
    """执行 SQL，返回字典列表"""
    if not _SAFE_PATTERN.match(sql):
        raise ValueError("只允许 SELECT 查询")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _get_schema() -> str:
    """返回数据库 Schema 描述，供 LLM 参考"""
    return """
数据库目前只有一张核心表：

financial_metrics（财务经营指标）：
  - company: 公司名（如"豪威集团"、"统一企业中国"、"索尼"）
  - ticker: 股票代码（如果没有则为 NULL）
  - metric: 标准化指标名（优先归一化为：收入/净利润/毛利率/净利率/市盈率/EV/EBITDA/市值/同比增速/市场份额；无法归类时保留原文指标名）
  - raw_metric: 原文指标表述（用于溯源核查）
  - is_estimate: 0=实际值 1=预测值
  - confidence: 数据权威度（1=研报, 2=官方财报, 3=终端数据）
  - source_file: 来源PDF文件名
  - source_page: 来源页码

# 💡 SQL 生成核心安全与准确性规则（必须严格遵守）：
1. 单表查询：所有数据（包括市场份额）均在 financial_metrics 表中，不要尝试查询其他不存在的表。
2. 年份模糊匹配：由于入库的年份格式可能存在微小差异（如 "2025" 或 "FY25"），当用户查询特定年份时，必须使用 LIKE 进行模糊查询。例如查询2025年，必须写成 `year LIKE '%25%'`。
3. 禁用复杂嵌套与行列转换：尽量使用简单的 SELECT ... WHERE ... 查询，严禁使用复杂的 CASE WHEN (Pivot操作) 将多行合并为单列。直接返回查到的多行原始记录即可。
4. 客观查询：除非用户明确提出要查询“预测”或“预期”数据，否则不要在 WHERE 语句中主动过滤 `is_estimate`，更不要自行在查询条件里给年份加 'E' 的后缀（如强行拼写 '%25E%'）。
5. 【溯源必选】：SELECT 语句必须始终包含 source_file 和 source_page，这是金融场景下数据可信度的核心依据，禁止省略。

查询示例：
  -- 例1：查询统一企业中国2025年的收入和净利润（含溯源）
  SELECT company, metric, value, unit, year, is_estimate, source_file, source_page
  FROM financial_metrics
  WHERE company LIKE '%统一%' AND metric IN ('收入', '净利润') AND year LIKE '%25%'
  ORDER BY confidence DESC, source_page ASC;

  -- 例2：对比所有公司2024年的毛利率（含溯源）
  SELECT company, metric, value, unit, year, source_file, source_page
  FROM financial_metrics
  WHERE metric = '毛利率' AND year LIKE '%24%'
  ORDER BY value DESC;
"""


@tool
def sql_tool(query: str) -> str:
    """
    查询金融研报结构化数据库，获取公司财务指标的精确数值。

    适用场景（必须优先于rag_tool调用）：
    - 查询特定公司某年份的具体财务数据（收入、净利润、毛利率、净利率、市盈率等）
    - 对比多家公司的同一财务指标
    - 查询某公司的市场份额数据
    - 按条件筛选或排名（如"毛利率最高的公司"）

    参数 query：用自然语言描述你想查询什么，工具会自动生成SQL并执行。
    例如："豪威集团2024年的毛利率"、"对比所有公司FY24的净利率"
    """
    sql = "（未能生成SQL）"
    try:
        schema = _get_schema()

        from react_agent.utils.llm import load_chat_model
        llm = load_chat_model(model_ref)

        sql_prompt = f"""根据以下数据库Schema，将自然语言查询转换为SQL。
只输出SQL语句，不要任何解释，不要markdown代码块。

Schema：
{schema}

查询：{query}

SQL："""

        sql_resp = llm.invoke(sql_prompt)
        sql = _extract_sql(sql_resp.content)

        rows = _query_db(sql)

        if not rows:
            return json.dumps(
                _ok(
                    tool_name=_TOOL_NAME,
                    query=query,
                    data={"result": "数据库中未找到相关数据", "rows": []},
                    meta={"sql": sql, "row_count": 0},
                ),
                ensure_ascii=False,
            )

        # 格式化结果，包含来源信息
        result_lines = []
        for row in rows:
            estimate_tag = "（预测值）" if row.get("is_estimate") == 1 else ""
            core = (
                f"{row.get('company')} | {row.get('metric')}{estimate_tag}"
                f" = {row.get('value')} {row.get('unit') or ''}"
                f" [{row.get('year')}]"
            )
            source_parts = []
            if row.get("source_file"):
                source_parts.append(row["source_file"])
            if row.get("source_page") is not None:
                source_parts.append(f"第 {row['source_page']} 页")
            source_str = f"\n  📄 来源: {' · '.join(source_parts)}" if source_parts else ""
            result_lines.append(core + source_str)

        return json.dumps(
            _ok(
                tool_name=_TOOL_NAME,
                query=query,
                data={"result": "\n".join(result_lines), "rows": rows},
                meta={"sql": sql, "row_count": len(rows)},
            ),
            ensure_ascii=False,
        )

    except ValueError as e:
        # SQL 安全拦截（非临时错误，不应重试）
        return json.dumps(
            _err(
                tool_name=_TOOL_NAME,
                query=query,
                message=str(e),
                code="BAD_INPUT",
            ),
            ensure_ascii=False,
        )

    except sqlite3.Error as e:
        # 数据库执行错误（SQL 语法错误、表不存在等）
        return json.dumps(
            _err(
                tool_name=_TOOL_NAME,
                query=query,
                message=f"数据库执行错误：{e}",
                code="DB_ERROR",
                meta={"sql": sql},
            ),
            ensure_ascii=False,
        )

    except Exception as e:
        # 兜底：LLM 调用失败、网络异常等
        return json.dumps(
            _err(
                tool_name=_TOOL_NAME,
                query=query,
                message=f"工具内部错误：{type(e).__name__}: {e}",
                code="TOOL_ERROR",
            ),
            ensure_ascii=False,
        )


# if __name__ == "__main__":
#     print("=== 开始测试 sql_tool ===")
#
#     # 这里的查询条件完美契合你刚刚入库的《大消费行业周报》第7页的数据
#     test_query = "给我查询一下 TikTokShop 的2024年市场份额是多少"
#
#     print(f"🗣️ 用户提问: {test_query}")
#     print("-" * 50)
#
#     try:
#         # 直接调用 tool 进行测试
#         result_json_str = sql_tool.invoke(test_query)
#
#         # 将返回的 JSON 字符串解析并格式化打印
#         result_dict = json.loads(result_json_str)
#         print("🤖 工具返回结果:")
#         print(json.dumps(result_dict, indent=2, ensure_ascii=False))
#
#         # 提取其中具体生成的 SQL 语句展示出来
#         if "meta" in result_dict and "sql" in result_dict["meta"]:
#             print("-" * 50)
#             print(f"🔍 LLM 实际执行的 SQL:\n{result_dict['meta']['sql']}")
#
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")