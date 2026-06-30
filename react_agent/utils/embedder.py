# react_agent/utils/embedder.py
from __future__ import annotations
import os
import threading
import torch
from langchain_huggingface import HuggingFaceEmbeddings

_embedder = None
_embedder_lock = threading.Lock()


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},  # bge 系列官方推荐
        )
    return _embedder