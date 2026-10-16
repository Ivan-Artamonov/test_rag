 Модель эмбеддингов

- Название: **BAAI/bge-m3**
- Репозиторий / API: https://huggingface.co/BAAI/bge-m3
- Размер эмбеддингов: **1024** измерения (dense)
- Загрузка через `sentence-transformers`; векторы нормализованы, в индексе
  используется косинусная метрика.

## База знаний

- Вымышленная вселенная (обфусцированный фандом по мотивам сериала
  "Silicon Valley"): персонажи, компании, эпизоды и понятия.
- 48 Markdown-файлов в папке `knowledge_base/` в корне репозитория.
- Тексты разбиты на логические чанки через `RecursiveCharacterTextSplitter`
  (100-300 слов), с сохранением источника и позиции.

# Индекс

- Векторная БД: **ChromaDB** (`PersistentClient`).
- Расположение: `software/chroma.db/`.
- Чанков в индексе: **208**.

# Время генерации

- Генерация эмбеддингов: **~7.8 c**
- Полное время построения индекса (с загрузкой модели): **~109 c**.

# Как воспроизвести

```bash
source .venv/bin/activate
python software/chunk_kb.py     # чанки
python software/build_index.py  # эмбеддинги + индекс
python software/query_index.py  # проверка поиска
```

# Примеры запросов: 
```
Открываю индекс chroma.db ...
Чанков в коллекции: 208, метрика: cosine
Загружаю модель BAAI/bge-m3 ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 51574.72it/s]

=== Запрос: 'What compression algorithm did Tobin Mercer invent?' ===

[1] близость=0.552
источник: optimal_tip_to_tip_efficiency.md  (заголовок='Optimal Tip-to-Tip Efficiency', позиция чанка=5)
Tobin is nervous and seems like he’s going to blow the presentation, but then he starts to explain Fernwright, saying that they used to be cloud-based, user-focused compression. He continues by saying that Omnicore did everything they were trying to do, so he changed what Fernwright is. He rebuilt the algorithm and tel...

[2] близость=0.507
источник: optimal_tip_to_tip_efficiency.md  (заголовок='Optimal Tip-to-Tip Efficiency', позиция чанка=2)
Back in the presentation hall, Aldous Vane is presenting Helix. Tobin assures Arjun that Helix can’t come close to Fernwright’s Kraus index, a measure of how well files compress. Aldous Vane reveals that Helix’ Kraus index is the best in the history of compression: 2.89, which happens to be the same as Fernwright’s. “A...

[3] близость=0.502
источник: omnicore.md  (заголовок='Omnicore', позиция чанка=2)
## History  ### Season One  In 2014, Omnicore employee Tobin Mercer created Fernwright and presented it to two Omnicore employees, who were originally dismissive towards Mercer' site which had been pitched as the "Gigabox of music" but were impressed by the level of compression done by the site. News of the compression...


=== Запрос: 'What is the Kraus index and why does it matter?' ===

[1] близость=0.668
источник: kraus_index.md  (заголовок='Kraus index', позиция чанка=0)
A Kraus index is a (fictional) test to see the efficiency of a compression algorithm. It was created by Kernbridge electrical engineering professor Amos Kraus and Ph.D. student Vikram Rao. It was believed that the theoretical limit for a Kraus index was 2.9 but this was disproved when the tech startup company Fernwrigh...

[2] близость=0.502
источник: optimal_tip_to_tip_efficiency.md  (заголовок='Optimal Tip-to-Tip Efficiency', позиция чанка=2)
Back in the presentation hall, Aldous Vane is presenting Helix. Tobin assures Arjun that Helix can’t come close to Fernwright’s Kraus index, a measure of how well files compress. Aldous Vane reveals that Helix’ Kraus index is the best in the history of compression: 2.89, which happens to be the same as Fernwright’s. “A...

[3] близость=0.497
источник: optimal_tip_to_tip_efficiency.md  (заголовок='Optimal Tip-to-Tip Efficiency', позиция чанка=5)
Tobin is nervous and seems like he’s going to blow the presentation, but then he starts to explain Fernwright, saying that they used to be cloud-based, user-focused compression. He continues by saying that Omnicore did everything they were trying to do, so he changed what Fernwright is. He rebuilt the algorithm and tel...


=== Запрос: 'Who is Aldous Vane and what is his role at Omnicore?' ===

[1] близость=0.681
источник: aldous_vane.md  (заголовок='Aldous Vane', позиция чанка=0)
Aldous Vane is a billionaire senior executive at Omnicore, and the main antagonist in Beacon's Circadia Valley.  ## Background  Aldous Vane was the Chief Innovation Officer of Omnicore, a huge tech company in Circadia Valley that seemingly has its hands in everything but specifically does “targeted search and cross-pla...

[2] близость=0.624
источник: minimum_viable_product.md  (заголовок='Minimum Viable Product', позиция чанка=0)
Minimum Viable Product  Gallery   Transcript  ▶️ Next episode — Episode info  Director: Mike Judge — Writers: Mike Judge, John Altschuler, Dave Krinsky — Aired: April 6, 2014 —Season: One — Number: One Summary: An engineer is stuck between a bidding war with Magnus Quill and Aldous Vane after showcasing his lossless co...

[3] близость=0.604
источник: omnicore.md  (заголовок='Omnicore', позиция чанка=0)
Omnicore  Purpose : Multinational Internet technologies company — Gallery —Location details  First appearance: "Minimum Viable Product"  Employees: Aldous Vane Visitors: Omnicore employees  What is Omnicore? Excellent question. Omnicore isn't just another high-tech company. Omnicore isn't just about software. Omnicore,...
```