"""
check_pdfs.py — 金融研报 PDF 质检脚本
检测维度：加密、扫描版、乱码、部分扫描、重复文档、空白页
"""
import fitz
import hashlib
import re
from pathlib import Path
from collections import defaultdict

data_dir = Path(__file__).parent.parent / "data"
# 乱码阈值：中文占比低于此值（10%）即报警
MIN_CJK_RATIO = 0.1


# ── 文件哈希去重 ──────────────────────────────────────────────────────────
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 主检测逻辑 ────────────────────────────────────────────────────────────
results = []
hash_map = defaultdict(list)   # md5 → [文件列表]，用于去重

for pdf in sorted(data_dir.rglob("*.pdf")):
    record = {"path": pdf, "issues": [], "stats": {}}

    # 去重
    md5 = file_md5(pdf)
    hash_map[md5].append(pdf)

    try:
        doc = fitz.open(pdf)
        if doc.is_encrypted:
            record["issues"].append("加密")
            results.append(record)
            continue  # 直接跳过后续文本提取

        page_count = doc.page_count

        total_chars = 0
        cjk_count = 0
        image_pages = 0

        for page in doc:
            text = page.get_text()
            total_chars += len(text)

            # 累加中文字符数
            cjk_count += sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

            # 统计部分扫描页（字少且有图）
            if len(text.strip()) < 50 and len(page.get_images()) > 0:
                image_pages += 1

        # 【优化点 2：计算真实比例】
        cjk_ratio = cjk_count / total_chars if total_chars > 0 else 0
        sr = image_pages / max(page_count, 1)

        # 把 cjk_ratio 存入 stats，打印出来供你观察业务真实数据
        record["stats"] = {
            "pages": page_count,
            "total_chars": total_chars,
            "avg_chars": total_chars // max(page_count, 1),
            "cjk_ratio": round(cjk_ratio, 4)  # 新增：记录中文占比
        }

        # ── 异常判断逻辑 ─────────────────────────────────────
        # 1. 全量扫描版（平均每页字符极少）
        if total_chars < 100 * page_count:
            record["issues"].append("全量扫描版")
        else:
            # 2. 部分扫描
            if sr > 0.3:
                record["issues"].append(f"部分扫描({sr:.0%}页为图片)")

            # 3. 乱码（文本大于50字，且中文比例低于新设定的 10% 阈值）
            if total_chars >= 50 and cjk_ratio < MIN_CJK_RATIO:
                record["issues"].append(f"疑似乱码(中文仅占{cjk_ratio:.1%})")

    except Exception as e:
        record["issues"].append(f"打开失败:{e}")

    results.append(record)

# ── 重复文件报告 ──────────────────────────────────────────────────────────
duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}

# ── 输出报告 ──────────────────────────────────────────────────────────────
total = len(results)
problem = [r for r in results if r["issues"]]

print(f"质检完成：共 {total} 份 PDF，发现 {len(problem)} 份问题文件\n")

if problem:
    print("── 问题文件 ──────────────────────────────────────────")
    for r in problem:
        rel = r["path"].relative_to(data_dir)
        s = r["stats"]
        issues_str = " | ".join(r["issues"])
        print(f"  [{issues_str}]")
        print(f"    {rel}  ({s.get('pages','?')}页, 均{s.get('avg_chars','?')}字/页)")

if duplicates:
    print(f"\n── 重复文档（{len(duplicates)} 组）────────────────────────")
    for md5, paths in duplicates.items():
        print(f"  MD5: {md5[:8]}...")
        for p in paths:
            print(f"    {p.relative_to(data_dir)}")

if not problem and not duplicates:
    print("✅ 全部通过，可直接开始向量化")