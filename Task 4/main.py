#!/usr/bin/env python3
"""консольный интерфейс к RAG

Позволяет общаться с ботом в цикле: на каждый вопрос бот ищет чанки в индексе, генерирует ответ (few-shot + CoT) и показывает источники, на которые опирался.

Запуск (из корня репозитория, с venv):
    .venv/bin/python "Task 4/main.py"

Внутри REPL команды выхода: exit, quit, q, выход.
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prompts
import rag

EXIT_COMMANDS = {"exit", "quit", "q", "выход", "стоп"}

def build_bot():
    print(f"Инициализация...")
    return rag.RagBotManual()


def print_sources(chunks):
    print("\nИсточники:")
    if not chunks:
        print("  (не найдено)")
        return
    for i, ch in enumerate(chunks, 1):
        print(f"  [{i}] {ch['source']} (чанк {ch['chunk_index']}), "
              f"близость {ch['similarity']:.3f}")


def repl(bot):
    print("=" * 60)
    print(f"RAG-бот по базе знаний")
    print("Введите вопрос или команду выхода (exit / quit / q).")
    print("=" * 60)
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Пока!")
            break

        answer, chunks = bot.answer(question)
        print("\nБот:")
        print(answer)
        print_sources(chunks)


def main():
    bot = build_bot()
    repl(bot)

if __name__ == "__main__":
    main()
