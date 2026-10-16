#!/usr/bin/env python3
"""ручной RAG-пайплайн

вопрос -> эмбеддинг (bge-m3) -> поиск в ChromaDB -> сборка промпта
(system + few-shot + CoT + найденные фрагменты) -> LLM (Qwen3-14B)
-> текст ответа.

Запуск (из корня репозитория, с venv): .venv/bin/python tasks/4/rag_manual.py "вопрос"
"""
import sys
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Пути считаем от расположения скрипта, чтобы не зависеть от cwd.
SCRIPT_DIR = Path(__file__).resolve().parent          # tasks/4
INDEX_DIR = "Task 3/software/chroma.db"  # индекс из Задания 3
sys.path.insert(0, str(SCRIPT_DIR))
import prompts  # noqa: E402  (общие промпты лежат рядом)

COLLECTION_NAME = "knowledge_base"
ENCODER_NAME = "BAAI/bge-m3"   # та же модель, что при индексации
MODEL_NAME = "qwen3:14b"        # LLM через Ollama
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TOP_K = 3
TEMPERATURE = 0.2               # низкая - меньше галлюцинаций для RAG Q&A


class RagBotManual:
    """энкодер + ChromaDB + HTTP-вызов Ollama."""

    def __init__(self):
        print(f"Загружаю энкодер {ENCODER_NAME} ...")
        self.encoder = SentenceTransformer(ENCODER_NAME)
        print(f"Открываю индекс {INDEX_DIR} ...")
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, question, top_k=TOP_K):
        """Эмбеддинг запроса -> поиск top-k ближайших чанков.

        Возвращает список словарей с полями text, source, chunk_index,
        similarity (косинусная близость = 1 - расстояние).
        """
        emb = self.encoder.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(res["ids"][0])):
            chunks.append({
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i]["source"],
                "chunk_index": res["metadatas"][0][i]["chunk_index"],
                "similarity": 1.0 - res["distances"][0][i],
            })
        return chunks

    def generate(self, question, chunks):
        """Собирает промпт (system + few-shot + CoT + контекст + вопрос) и
        отправляет его в Ollama через HTTP API."""
        user_msg = prompts.USER_TEMPLATE.format(
            context=prompts.format_context(chunks),
            question=question,
        )
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,                      # отключаем встроенный thinking qwen3,
                                                 # CoT задаётся явно в system-промпте
            "options": {"temperature": TEMPERATURE},
        }
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def answer(self, question, top_k=TOP_K):
        """Оркестрация: найти чанки -> порог -> генерация ответа.

        Возвращает кортеж (ответ, чанки). Чанки нужны вызывающему коду,
        чтобы показать источники. Если лучший чанк ниже порога близости -
        отвечаем "Я не знаю" без обращения к LLM.
        """
        chunks = self.retrieve(question, top_k)
        if not chunks or chunks[0]["similarity"] < prompts.SIMILARITY_THRESHOLD:
            return prompts.DONT_KNOW_ANSWER, chunks
        return self.generate(question, chunks), chunks


def main():
    if len(sys.argv) < 2:
        print("Использование: rag.py \"вопрос\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    bot = RagBotManual()
    answer, chunks = bot.answer(question)
    print("\n=== Найденные чанки ===")
    for i, ch in enumerate(chunks, 1):
        print(f"[{i}] {ch['source']} (чанк {ch['chunk_index']}), "
              f"близость {ch['similarity']:.3f}")
    print("\n=== Ответ ===")
    print(answer)


if __name__ == "__main__":
    main()
