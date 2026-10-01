#!/usr/bin/env python3
"""пример семантического поиска по построенному индексу.

Открывает индекс ChromaDB (software/chroma.db), для каждого тестового
запроса генерирует эмбеддинг той же моделью bge-m3 и возвращает top-k чанков
с метаданными источника (title, source, chunk_index) - то, на что потом будет
опираться бот для цитирования.

Запуск (из корня репозитория, с venv):
    .venv/bin/python software/query_index.py
    .venv/bin/python software/query_index.py "свой запрос"
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

SOFTWARE_DIR = Path(__file__).resolve().parent
INDEX_DIR = SOFTWARE_DIR / "chroma.db"
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "BAAI/bge-m3"

TOP_K = 3
SNIPPET_CHARS = 320

# 2-3 тестовых запроса: технология, метрика, персонаж.
DEFAULT_QUERIES = [
    "What compression algorithm did Tobin Mercer invent?",
    "What is the Kraus index and why does it matter?",
    "Who is Aldous Vane and what is his role at Omnicore?",
]


def search(query: str, model, collection, top_k: int = TOP_K):
    emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return res


def print_results(query: str, res):
    print(f"\n=== Запрос: {query!r} ===\n")
    for i in range(len(res["ids"][0])):
        doc = res["documents"][0][i]
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        sim = 1.0 - dist                     # cosine: близость = 1 - расстояние
        snippet = doc.replace("\n", " ").strip()
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."
        print(f"[{i + 1}] близость={sim:.3f}")
        print(f"    источник: {meta['source']}  (заголовок={meta['title']!r}, "
              f"позиция чанка={meta['chunk_index']})")
        print(f"    {snippet}\n")


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES

    print(f"Открываю индекс {INDEX_DIR} ...")
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Чанков в коллекции: {collection.count()}, "
          f"метрика: {collection.metadata.get('hnsw:space', 'l2')}")

    print(f"Загружаю модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    for q in queries:
        res = search(q, model, collection)
        print_results(q, res)


if __name__ == "__main__":
    main()
