from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """
    API 层配置（pydantic-settings，从环境变量 / .env 读取）。

    设计约定：本文件零副作用 —— 只定义类，不做 sys.exit()、不实例化单例。
    实例化 + fail-fast 校验在 main.py 的 app 构造阶段执行，
    确保 pytest import 本模块时不会因缺少环境变量而杀死测试进程。
    """

    cors_allow_origins: list[str] = Field(default=["*"])
    api_key: Optional[str] = Field(default=None)
    rate_limit_rpm: int = Field(default=60, gt=0)
    stream_timeout_s: int = Field(default=120, gt=0)
    invoke_timeout_s: int = Field(default=60, gt=0)
    daily_token_budget: int = Field(
        default=1_000_000,
        gt=0,
        description="每个 API Key 每日 token 上限（默认 100 万）。超限返回 429。设为极大值可禁用。",
    )

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def validate_llm_key(self) -> None:
        """
        Provider-aware LLM key validation。

        从 agent settings 读取当前模型字符串（格式 "provider/model-name"），
        提取 provider 前缀，仅校验该 provider 对应的 key。
        - "local" provider 无外部 key 要求，直接跳过。
        - 未知 provider 宽容跳过（避免新 provider 接入时阻断启动）。

        由 main.py 在 app 构造前显式调用，不在 import 时自动执行。
        """
        from react_agent.core.config import settings as _s

        model: str = (_s.llm.model or "").strip()
        provider = model.split("/")[0].lower() if "/" in model else model.lower()

        if provider in ("local", ""):
            return

        _key_map: dict[str, tuple[str, str]] = {
            "deepseek":  ("DEEPSEEK_API_KEY", _s.secrets.DEEPSEEK_API_KEY),
            "openai":    ("OPENAI_API_KEY",   _s.secrets.OPENAI_API_KEY),
            "anthropic": ("OPENAI_API_KEY",   _s.secrets.OPENAI_API_KEY),
        }

        if provider not in _key_map:
            return

        env_name, key_value = _key_map[provider]
        if not key_value:
            raise ValueError(
                f"模型 provider='{provider}'，但 {env_name} 未设置。\n"
                f"请在 .env 文件中添加: {env_name}=<your-key>"
            )
