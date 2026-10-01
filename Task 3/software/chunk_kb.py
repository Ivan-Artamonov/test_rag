#!/usr/bin/env python3
"""разбиение базы знаний (knowledge_base/*.md) на чанки.

Читает все .md-файлы вымышленной вселенной, режет каждый документ на
логически связные чанки через RecursiveCharacterTextSplitter (LangChain)
и сохраняет манифест chunks.json с текстом и метаданными (источник,
заголовок, позиция, id).

Запуск (из корня репозитория, с venv):
    .venv/bin/python software/chunk_kb.py
"""
import json
import os
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Пути считаем от расположения скрипта, чтобы не зависеть от cwd.
SOFTWARE_DIR = Path(__file__).resolve().parent          # tasks/3/software
REPO_ROOT = SOFTWARE_DIR.parents[2]                     # корень репозитория
KB_DIR = REPO_ROOT / "knowledge_base"
MANIFEST_PATH = SOFTWARE_DIR / "chunks.json"

# Параметры нарезки.
CHUNK_SIZE = 1000       # символов (~170 слов, ~230 токенов - в диапазоне задания)
CHUNK_OVERLAP = 150     # перекрытие между соседними чанками (сохраняет контекст)
SEPARATORS = ["\n\n", "\n", " ", ""]  # режем сначала по абзацам/секциям


def slugify(title: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", title).strip("_").lower()
    return s or "page"


def load_documents(kb_dir: Path):
    """Возвращает список (title, path, text) для каждого .md-файла.

    Заголовок H1 (`# Title`) убирается из текста и кладётся в метаданные,
    чтобы не создавать вырожденный чанк из одного заголовка. Заголовки
    секций `##`/`###` остаются в тексте (дают контекст чанку).
    """
    docs = []
    for p in sorted(kb_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8").strip()
        title = p.stem  # запасной вариант
        m = re.match(r"#\s+(.+)\n?", text)
        if m:
            title = m.group(1).strip()
            text = text[m.end():].strip()
        docs.append((title, str(p), text))
    return docs


def chunk_knowledge_base(kb_dir: Path = KB_DIR):
    """Разбивает базу знаний на чанки с метаданными."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = []
    for title, path, text in load_documents(kb_dir):
        base = slugify(title)
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{base}_{i}",
                "title": title,
                "source": os.path.basename(path),
                "chunk_index": i,
                "text": piece,
            })
    return chunks


def main():
    chunks = chunk_knowledge_base()
    sizes_chars = sorted(len(c["text"]) for c in chunks)
    sizes_words = sorted(len(c["text"].split()) for c in chunks)
    total_chars = sum(sizes_chars)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Документов (файлов .md): {len(set(c['source'] for c in chunks))}")
    print(f"Чанков всего:            {len(chunks)}")
    print(f"Символов всего:          {total_chars}")
    print(f"Размер чанка (символы):  min={sizes_chars[0]} "
          f"медиана={sizes_chars[len(sizes_chars) // 2]} max={sizes_chars[-1]}")
    print(f"Размер чанка (слова):    min={sizes_words[0]} "
          f"медиана={sizes_words[len(sizes_words) // 2]} max={sizes_words[-1]}")
    print(f"Манифест:                {MANIFEST_PATH}")

    # Пример для наглядности.
    print("\n--- Пример чанка ---")
    ex = next(c for c in chunks if len(c["text"].split()) > 30)
    print(f"id={ex['id']}  source={ex['source']}  "
          f"title={ex['title']!r}  chunk_index={ex['chunk_index']}")
    print(ex["text"][:400] + ("..." if len(ex["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
