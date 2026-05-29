import os
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import settings
from app.prompts import build_system_prompt, parse_and_strip_profile_tags
from app import fewshot

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.openai_api_key,
)
_vectorstore: FAISS | None = None
_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=settings.openai_api_key,
)


def _load_vectorstore() -> None:
    global _vectorstore
    index_path = settings.faiss_index_path
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        _vectorstore = FAISS.load_local(
            index_path,
            _embeddings,
            allow_dangerous_deserialization=True,
        )


def search(query: str, k: int = 3) -> Tuple[List[str], float]:
    """Returns (chunks, best_distance). best_distance < threshold means relevant."""
    if _vectorstore is None:
        return [], 1.0

    docs_and_scores = _vectorstore.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        return [], 1.0

    chunks = [doc.page_content for doc, _ in docs_and_scores]
    best_score = float(docs_and_scores[0][1])
    return chunks, best_score


async def get_answer(
    user_message: str,
    history: List[dict],
    profile: dict = None,
) -> Tuple[str, dict]:
    """Returns (clean_reply, profile_updates)."""
    chunks, best_score = search(user_message)
    examples = fewshot.get_examples(user_message)

    system_text, trimmed_history = build_system_prompt(
        rag_chunks=chunks,
        fewshot_examples=examples,
        history=history,
        user_message=user_message,
        profile=profile or {},
    )

    messages = [SystemMessage(content=system_text)]
    for msg in trimmed_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    response = await _llm.ainvoke(messages)
    clean_reply, profile_updates = parse_and_strip_profile_tags(response.content)
    return clean_reply, profile_updates


_load_vectorstore()
