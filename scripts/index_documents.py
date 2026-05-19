"""Build FAISS index from product docs (.md / .docx)."""
import re
from pathlib import Path

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
import os

load_dotenv()

DOCS_DIR = Path("data/product_docs")
INDEX_DIR = Path("data/faiss_index")

# Lines to strip from the docx (slide markers and similar noise)
NOISE_RE = re.compile(
    r"слайд\s+начинается|слайд\s+заканчивается|slide\s+start|slide\s+end"
    r"|^\s*https?://\S+\s*$",  # bare URLs on their own line
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line for line in lines if not NOISE_RE.match(line.strip())]
    return "\n".join(cleaned)


def load_documents() -> list[Document]:
    docs = []
    for path in DOCS_DIR.iterdir():
        if path.suffix == ".md":
            loader = TextLoader(str(path), encoding="utf-8")
        elif path.suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            continue
        print(f"Загружаю: {path.name}")
        loaded = loader.load()
        for doc in loaded:
            doc.page_content = clean_text(doc.page_content)
        docs.extend(loaded)
    return docs


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан в .env")

    documents = load_documents()
    if not documents:
        print(f"Документы не найдены в {DOCS_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Drop empty chunks after cleaning
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Чанков после очистки: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Индекс сохранён в: {INDEX_DIR}")


if __name__ == "__main__":
    main()
