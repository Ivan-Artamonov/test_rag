#!/usr/bin/env python3
"""эмбеддинги чанков и построение векторного индекса в ChromaDB.

Читает манифест chunks.json (результат этапа B), прогоняет каждый чанк через
эмбеддинг-модель BAAI/bge-m3 (выбрана в Задании 1) и складывает векторы с
метаданными (источник, заголовок, позиция) в ChromaDB (PersistentClient).
Индекс сохраняется на диск в папке tasks/3/software/chroma.db.

Запуск (из корня репозитория, с venv):
    .venv/bin/python software/build_index.py
"""
import json
import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Пути считаем от расположения скрипта, чтобы не зависеть от cwd.
SOFTWARE_DIR = Path(__file__).resolve().parent       # tasks/3/software
MANIFEST_PATH = SOFTWARE_DIR / "chunks.json"
INDEX_DIR = SOFTWARE_DIR / "chroma.db"               # PersistentClient хранит здесь
COLLECTION_NAME = "knowledge_base"

MODEL_NAME = "BAAI/bge-m3"   # dense-эмбеддинги размерности 1024
BATCH_SIZE = 16              # сколько чанков кодировать за один проход


def load_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    t0 = time.time()

    chunks = load_chunks(MANIFEST_PATH)
    texts = [c["text"] for c in chunks]
    print(f"Чанков загружено: {len(chunks)}")

    # Загрузка модели (при первом запуске скачивает веса с HuggingFace).
    print(f"Загружаю модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()
    print(f"Размерность эмбеддингов: {dim}")

    # Генерация эмбеддингов (нормализуем -> работает косинусное расстояние).
    print("Генерирую эмбеддинги ...")
    t_emb = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_secs = time.time() - t_emb
    print(f"Эмбеддинги готовы за {emb_secs:.1f} c "
          f"(штук: {len(embeddings)}, форма: {embeddings.shape})")

    # Индекс в ChromaDB (PersistentClient - файлы на диске, без сервера).
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)   # пересобираем начисто
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},          # bge-m3 обучалась под cosine
    )

    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "title": c["title"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    print("Пишу векторы в ChromaDB ...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    total_secs = time.time() - t0
    n_docs = len(set(c["source"] for c in chunks))
    print("\n=== Индекс построен ===")
    print(f"Модель:            {MODEL_NAME} (dense, {dim} измерения)")
    print(f"Документов:        {n_docs}")
    print(f"Чанков в индексе:  {collection.count()}")
    print(f"Время эмбеддингов: {emb_secs:.1f} c")
    print(f"Время всего:       {total_secs:.1f} c")
    print(f"Индекс сохранён:   {INDEX_DIR}")


if __name__ == "__main__":
    main()
