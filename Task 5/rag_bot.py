#!/usr/bin/env python3
"""4: ручной пайплайн + защита от инъекций.

Повторяет ручной пайплайн из Задания 4, но с двумя отличиями:

1. Индекс - копия chroma.db, куда добавлен вредоносный документ.
2. В answer() добавлен параметр protect - уровень защиты из safety.py ("none", "preprompt", "filter", "sanitize").

Запуск (из корня репозитория, с venv): .venv/bin/python tasks/5/rag_bot.py "вопрос" --protect filter
"""
import sys
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS_DIR = SCRIPT_DIR.parent
INDEX_DIR = SCRIPT_DIR / "chroma.db"

# Общие промпты из Задания 4 + модуль защиты (лежит рядом).
sys.path.insert(0, str(TASKS_DIR / "Task 4"))
import prompts
import safety

COLLECTION_NAME = "knowledge_base"
ENCODER_NAME = "BAAI/bge-m3"
MODEL_NAME = "qwen3:14b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TOP_K = 3
TEMPERATURE = 0.2

PROTECTION_LEVELS = ("none", "preprompt", "filter", "sanitize")


class RagBot:
    """Ручной RAG-пайплайн с настраиваемым уровнем защиты."""

    def __init__(self):
        print(f"Загружаю энкодер {ENCODER_NAME} ...")
        self.encoder = SentenceTransformer(ENCODER_NAME)
        print(f"Открываю индекс (копия) {INDEX_DIR} ...")
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, question, top_k=TOP_K):
        """Эмбеддинг запроса -> top-k ближайших чанков (косинусная близость)."""
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

    def _system_prompt(self, protect):
        """System-промпт с учётом уровня защиты (preprompt добавляет addon)."""
        system = prompts.SYSTEM_PROMPT
        if protect == "preprompt":
            system += "\n\n" + safety.SAFETY_SYSTEM_ADDON
        return system

    def generate(self, question, chunks, protect="none"):
        """Собирает промпт и отправляет его в Ollama (HTTP API)."""
        user_msg = prompts.USER_TEMPLATE.format(
            context=prompts.format_context(chunks),
            question=question,
        )
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": self._system_prompt(protect)},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": TEMPERATURE},
        }
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def answer(self, question, top_k=TOP_K, protect="none"):
        """Оркестрация: поиск -> защита -> порог -> генерация.

        Возвращает кортеж (answer, kept, dropped):
        - answer  - текст ответа ("Я не знаю.", если ниже порога);
        - kept    - чанки, попавшие в LLM (после защиты);
        - dropped - чанки, отброшенные фильтром (для показа в логе).
        """
        chunks = self.retrieve(question, top_k)
        kept, dropped = safety.apply_chunk_protection(protect, chunks)
        if not kept or kept[0]["similarity"] < prompts.SIMILARITY_THRESHOLD:
            return prompts.DONT_KNOW_ANSWER, kept, dropped
        return self.generate(question, kept, protect=protect), kept, dropped


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG-бот (Задание 5)")
    parser.add_argument("question", nargs="+", help="вопрос")
    parser.add_argument("--protect", choices=PROTECTION_LEVELS, default="none")
    args = parser.parse_args()

    bot = RagBot()
    question = " ".join(args.question)
    answer, kept, dropped = bot.answer(question, protect=args.protect)

    print("\n=== Найденные чанки (после защиты) ===")
    for i, ch in enumerate(kept, 1):
        print(f"[{i}] {ch['source']} (чанк {ch['chunk_index']}), "
              f"близость {ch['similarity']:.3f}")
    if dropped:
        print("\n=== Отфильтровано ===")
        for i, ch in enumerate(dropped, 1):
            print(f"[{i}] {ch['source']} (чанк {ch['chunk_index']}) - "
                  f"вредоносный, отброшен")
    print(f"\n=== Ответ (защита: {args.protect}) ===")
    print(answer)


if __name__ == "__main__":
    main()
