"""
B7 — Prometheus 指标验证
- 流式请求后 llm_tokens_total / llm_cost_usd_total 有值
- rpm 打爆后 rate_limit_rejections_total{reason="rpm"} 增加
- 断连后 in-flight 回 0（以稳定 1 = scrape 自身为验证点）
- path label 是路由模板 /api/v1/chat/stream，不是 unmatched
"""
from __future__ import annotations

import re
import pytest

pytestmark = pytest.mark.asyncio


def _get_metric(metrics_text: str, name: str) -> list[tuple[dict, float]]:
    """
    从 metrics 文本中提取指定 metric 的所有 {labels} value 对。
    返回 [(label_dict, value), ...]
    """
    results = []
    pattern = re.compile(
        rf'^{re.escape(name)}'
        r'\{([^}]*)\}\s+([\d.e+\-]+)',
        re.MULTILINE,
    )
    for m in pattern.finditer(metrics_text):
        raw_labels = m.group(1)
        value = float(m.group(2))
        labels: dict = {}
        for part in raw_labels.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
        results.append((labels, value))
    return results


async def test_llm_tokens_recorded_after_stream(client, fake_redis):
    """流式请求完成后，llm_tokens_total 有非零值。"""
    r = await client.post("/api/v1/chat/stream", json={"message": "hi"})
    assert r.status_code == 200

    metrics = (await client.get("/api/v1/metrics")).text
    token_entries = _get_metric(metrics, "llm_tokens_total")

    assert token_entries, "llm_tokens_total should have entries after a stream request"
    total = sum(v for _, v in token_entries)
    assert total > 0, f"llm_tokens_total sum should be > 0, got {total}"


async def test_llm_cost_recorded_after_stream(client, fake_redis):
    """流式请求完成后，llm_cost_usd_total 有非零值。"""
    r = await client.post("/api/v1/chat/stream", json={"message": "hi"})
    assert r.status_code == 200

    metrics = (await client.get("/api/v1/metrics")).text
    cost_entries = _get_metric(metrics, "llm_cost_usd_total")

    assert cost_entries, "llm_cost_usd_total should have entries"
    total_cost = sum(v for _, v in cost_entries)
    assert total_cost > 0, f"llm_cost_usd_total should be > 0, got {total_cost}"


async def test_rate_limit_rejection_counter(client, monkeypatch, fake_redis):
    """rpm 超限后 rate_limit_rejections_total{reason='rpm'} 增加。"""
    import api.settings as _s
    class _Settings(_s.APISettings):
        api_key: None = None
        rate_limit_rpm: int = 1
        daily_token_budget: int = 1_000_000
    monkeypatch.setattr(_s, "APISettings", _Settings)

    # 拿超限前的计数
    metrics_before = (await client.get("/api/v1/metrics")).text
    before_entries = _get_metric(metrics_before, "rate_limit_rejections_total")
    before_rpm = sum(v for lbl, v in before_entries if lbl.get("reason") == "rpm")

    # 触发 rpm 超限
    await client.post("/api/v1/chat/stream", json={"message": "x"})
    r2 = await client.post("/api/v1/chat/stream", json={"message": "x"})
    assert r2.status_code == 429

    metrics_after = (await client.get("/api/v1/metrics")).text
    after_entries = _get_metric(metrics_after, "rate_limit_rejections_total")
    after_rpm = sum(v for lbl, v in after_entries if lbl.get("reason") == "rpm")

    assert after_rpm > before_rpm, (
        f"rate_limit_rejections_total{{reason='rpm'}} should increase: "
        f"{before_rpm} → {after_rpm}"
    )


async def test_http_path_label_is_route_template(client, fake_redis):
    """
    http_requests_total 的 path label 必须是路由模板（/api/v1/chat/stream），
    不能是原始 URL（含 session_id）。
    注：404 请求合法产生 'unmatched' label，不在此断言范围内。
    这验证 endpoint→path 反向映射正确工作（Starlette 0.52.1 不设 scope['route']）。
    """
    await client.post("/api/v1/chat/stream", json={"message": "hi"})

    metrics = (await client.get("/api/v1/metrics")).text
    req_entries = _get_metric(metrics, "http_requests_total")
    paths = {lbl.get("path") for lbl, _ in req_entries}

    assert "/api/v1/chat/stream" in paths, (
        f"Expected '/api/v1/chat/stream' in path labels, got: {paths}"
    )
    # 验证 POST /api/v1/chat/stream 有明确的路由模板 label（非 unmatched）
    stream_entries = [
        (lbl, v) for lbl, v in req_entries
        if lbl.get("path") == "/api/v1/chat/stream" and lbl.get("method") == "POST"
    ]
    assert stream_entries, (
        "POST /api/v1/chat/stream should appear with route template path label, "
        f"indicating reverse map works. All paths seen: {paths}"
    )


async def test_inflight_stable_after_requests(client, fake_redis):
    """
    in-flight 在所有请求结束后应回到 0（无泄漏）。

    注：真实 HTTP 服务器上，scrape 自身会让 gauge=1；
    但 httpx ASGITransport 的请求生命周期是原子的——generate_latest() 被调用时
    本次请求已完成（gauge 已 dec），因此在测试环境中观测到 0，而非 1。
    这里验证无泄漏（连续两次 scrape 均稳定），不依赖自计数行为。
    """
    # 先跑几条请求
    for _ in range(2):
        await client.post("/api/v1/chat/stream", json={"message": "test"})

    m1 = (await client.get("/api/v1/metrics")).text
    m2 = (await client.get("/api/v1/metrics")).text

    inflight_entries_1 = _get_metric(m1, "http_requests_in_flight")
    inflight_entries_2 = _get_metric(m2, "http_requests_in_flight")

    v1 = inflight_entries_1[0][1] if inflight_entries_1 else 0.0
    v2 = inflight_entries_2[0][1] if inflight_entries_2 else 0.0

    # 两次 scrape 均稳定（不增长），证明无泄漏
    assert v1 == v2, f"in-flight should be stable (no leak): {v1} vs {v2}"
    assert v1 == 0.0, f"After all requests done, in-flight should be 0 (no leak), got {v1}"


async def test_metrics_excludes_self_from_request_total(client, fake_redis):
    """/api/v1/metrics 自身不出现在 http_requests_total 中（避免自计数）。"""
    # 多次 scrape
    for _ in range(3):
        await client.get("/api/v1/metrics")

    metrics = (await client.get("/api/v1/metrics")).text
    req_entries = _get_metric(metrics, "http_requests_total")
    paths = {lbl.get("path") for lbl, _ in req_entries}

    assert "/api/v1/metrics" not in paths, (
        "Metrics scrape endpoint should be excluded from http_requests_total"
    )
