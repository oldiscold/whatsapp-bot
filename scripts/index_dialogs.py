"""Build FAISS index for few-shot examples from fewshot_pairs.json."""
import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

PAIRS_FILE = Path("data/fewshot_pairs.json")
INDEX_DIR = Path("data/fewshot_index")
BATCH_SIZE = 100


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан в .env")

    if not PAIRS_FILE.exists():
        print(f"Файл {PAIRS_FILE} не найден. Сначала запустите parse_dialogs.py")
        return

    with open(PAIRS_FILE, encoding="utf-8") as f:
        pairs = json.load(f)

    if not pairs:
        print("Нет пар для индексации")
        return

    print(f"Индексирую {len(pairs)} пар...")
    client = OpenAI(api_key=api_key)

    queries = [p["client"] for p in pairs]
    all_embeddings = []

    for i in range(0, len(queries), BATCH_SIZE):
        batch = queries[i : i + BATCH_SIZE]
        vecs = embed_batch(client, batch)
        all_embeddings.extend(vecs)
        print(f"  Обработано: {min(i + BATCH_SIZE, len(queries))}/{len(queries)}")

    matrix = np.array(all_embeddings, dtype="float32")
    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    print(f"Индекс сохранён в: {INDEX_DIR}/index.faiss")


if __name__ == "__main__":
    main()
