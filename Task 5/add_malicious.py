#!/usr/bin/env python3
"""2: добавляем вредоносный документ в КОПИЮ индекса.

Читает malicious_doc.md, кодирует его тем же энкодером, что и при индексации, и добавляет чанк в коллекцию knowledge_base копии индекса Task 5/chroma.db.

Запуск: .venv/bin/python tasks/5/add_malicious.py
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DIR = SCRIPT_DIR / "chroma.db"
MALICIOUS_PATH = SCRIPT_DIR / "malicious_doc.md" # вредоносный документ

COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "BAAI/bge-m3"
CHUNK_ID = "malicious_doc_0"

def main():
    text = MALICIOUS_PATH.read_text(encoding="utf-8").strip()
    title = MALICIOUS_PATH.stem
    source = MALICIOUS_PATH.name
    print(f"Вредоносный текст: {text!r}")

    print(f"Загружаю модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Кодирую чанк (normalize_embeddings=True) ...")
    emb = model.encode([text], normalize_embeddings=True).tolist()

    print(f"Открываю копию индекса {INDEX_DIR} ...")
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    before = collection.count()

    # upsert: повторный запуск не дублирует чанк, а обновляет его.
    collection.upsert(
        ids=[CHUNK_ID],
        embeddings=emb,
        documents=[text],
        metadatas=[{"title": title, "source": source, "chunk_index": 0}],
    )

    print(f"\n=== Добавлено ===")
    print(f"Чанков в коллекции: было {before} -> стало {collection.count()}")

    # Проверка: чанк доступен для поиска.
    print("\n=== Проверка поиска (запрос 'swordfish') ===")
    q_emb = model.encode(["swordfish"], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        sim = 1.0 - res["distances"][0][i]
        print(f"[{i}] {meta['source']} (чанк {meta['chunk_index']}), "
              f"близость {sim:.3f}")
        print(f"    текст: {res['documents'][0][i]!r}")


if __name__ == "__main__":
    main()
