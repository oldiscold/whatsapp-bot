import json
import os
from typing import List, Dict

import faiss
import numpy as np
from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)
_index: faiss.Index | None = None
_pairs: List[Dict[str, str]] = []


def _load() -> None:
    global _index, _pairs
    index_file = os.path.join(settings.fewshot_index_path, "index.faiss")
    pairs_file = settings.fewshot_pairs_path

    if not os.path.exists(index_file) or not os.path.exists(pairs_file):
        return

    _index = faiss.read_index(index_file)
    with open(pairs_file, encoding="utf-8") as f:
        _pairs = json.load(f)


def _embed(text: str) -> np.ndarray:
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = np.array(response.data[0].embedding, dtype="float32")
    return vec / np.linalg.norm(vec)


def get_examples(query: str, k: int = 3) -> List[Dict[str, str]]:
    if _index is None or not _pairs:
        return []

    vec = _embed(query).reshape(1, -1)
    k = min(k, len(_pairs))
    _, indices = _index.search(vec, k)
    return [_pairs[i] for i in indices[0] if i < len(_pairs)]


_load()
