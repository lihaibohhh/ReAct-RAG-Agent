# SNAPSHOT: (2026-01-31-14:26) → 修改于 config 统一化重构
"""
react_agent/utils/llm.py

职责：
- 解析 Context.model（"provider/model-name"）
- 根据 provider 加载对应的 Chat Model
- 推理参数统一从 settings 读取，不再散写 os.getenv()

修改说明：
- ChatModelSettings 改为从 settings.llm 读取，移除 os.getenv 直接调用
- _require_env 改为从 settings.secrets 读取，移除 os.getenv 直接调用
- 对外接口 load_chat_model() 不变，graph.py 无需任何修改
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from langchain_core.language_models.chat_models import BaseChatModel

# ✅ 唯一改动：从 config 读，不再 import os + os.getenv
from react_agent.core.config import settings


# -----------------------------
# 统一的"默认推理参数策略"
# -----------------------------

@dataclass(frozen=True)
class ChatModelSettings:
    """
    推理参数现在全部从 settings.llm 读取。
    .env / config.yaml / Conda 环境变量的优先级由 config.py 统一处理。
    """
    temperature: float = None   # type: ignore  # 实际值由 __post_init__ 填充
    max_tokens:  int = None   # type: ignore
    timeout:     int = None   # type: ignore
    retries:     int = None   # type: ignore

    def __new__(cls):
        # dataclass frozen=True 不支持 __init__ 赋值，用 __new__ + object.__setattr__
        obj = object.__new__(cls)
        llm = settings.llm
        object.__setattr__(obj, "temperature", llm.llm_temperature)
        object.__setattr__(obj, "max_tokens",  llm.llm_max_tokens)
        object.__setattr__(obj, "timeout",     llm.llm_timeout)
        object.__setattr__(obj, "retries",     llm.llm_retries)
        return obj


def _parse_model_ref(model_ref: str) -> Tuple[str, str]:
    """
    解析形如 "provider/model-name" 的字符串。
    """
    model_ref = (model_ref or "").strip()
    if not model_ref:
        raise ValueError("Context.model 不能为空，例如：'openai/gpt-4.1-mini'")

    if "/" not in model_ref:
        raise ValueError(
            "Context.model 必须是 'provider/model-name' 格式，例如："
            "'anthropic/claude-sonnet-4-5' 或 'openai/gpt-4.1-mini' 或 'local/qwen2.5-7b'"
        )
    provider, model_name = model_ref.split("/", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not provider or not model_name:
        raise ValueError(
            "Context.model 格式不正确，应该是 'provider/model-name'，例如 'openai/gpt-4.1-mini'"
        )
    return provider, model_name


def _get_secret(attr: str, env_key: str, hint: str) -> str:
    """
    从 settings.secrets 取密钥；缺失时给出明确错误。
    替代原来的 _require_env(os.getenv(...))。
    """
    val = getattr(settings.secrets, attr, "").strip()
    if not val:
        raise EnvironmentError(
            f"缺少密钥 {env_key}。{hint}\n"
            f"请在 .env 文件 或 conda env config vars set {env_key}=... 中设置"
        )
    return val


# -----------------------------
# provider -> ChatModel 工厂
# -----------------------------

def _build_openai(model_name: str, s: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e

    return ChatOpenAI(
        model=model_name,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        max_retries=s.retries,
        # langchain_openai 会自动读取 OPENAI_API_KEY 环境变量
        # load_dotenv 已在 config.py 里写入 os.environ，此处无需显式传递
    )


def _build_anthropic(model_name: str, s: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-anthropic") from e

    return ChatAnthropic(
        model=model_name,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        max_retries=s.retries,
    )


def _build_local_openai_compatible(model_name: str, s: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e

    base_url = settings.secrets.LOCAL_OPENAI_BASE_URL or "http://127.0.0.1:8000/v1"
    api_key = settings.secrets.LOCAL_OPENAI_API_KEY or "local-key"

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        max_retries=s.retries,
    )


def _build_deepseek_openai_compatible(model_name: str, s: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("未安装依赖：pip install langchain-openai") from e

    base_url = _get_secret(
        "DEEPSEEK_BASE_URL", "DEEPSEEK_BASE_URL",
        "例如：DEEPSEEK_BASE_URL=https://api.deepseek.com"
    )
    api_key = _get_secret(
        "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY",
        "例如：DEEPSEEK_API_KEY=sk-..."
    )

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        max_retries=s.retries,
    )


# -----------------------------
# 对外唯一入口：load_chat_model
# -----------------------------

@lru_cache(maxsize=32)
def load_chat_model(model_ref: str) -> BaseChatModel:
    """
    graph.py 唯一依赖的函数，签名不变。
    """
    provider, model_name = _parse_model_ref(model_ref)
    s = ChatModelSettings()

    if provider == "openai":
        return _build_openai(model_name, s)

    if provider == "anthropic":
        return _build_anthropic(model_name, s)

    if provider in ("local", "qwen-local", "openai-compatible"):
        return _build_local_openai_compatible(model_name, s)

    if provider in ("deepseek", "ds"):
        return _build_deepseek_openai_compatible(model_name, s)

    raise ValueError(
        f"不支持的 provider：{provider}。支持：openai / anthropic / local / deepseek\n"
        f"你传入的是：{model_ref}"
    )