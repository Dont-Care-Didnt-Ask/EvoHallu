# Перенос формул на Gemma и OLMo

Эксперимент проверяет 25 существующих формул из `found_formulas.py` на признаках трёх моделей: Llama 3.1 8B, Gemma 4 E4B-it и OLMo 3 7B Instruct. Утверждения и метки одинаковые; меняются checkpoint, токенизатор и chat-шаблон. Генерации ответов, обучения и повторного поиска формул нет.

Это структурированная версия кода завершённого прогона от **22 октября 2026**. Численные результаты исходного эксперимента находятся в [results/](results/). Проверки соответствия рефакторинга исходному расчёту записаны в [validation.json](validation.json). Полный GPU-прогон после рефакторинга не повторялся; отдельно проверены ядро признаков, токенизация и CPU-расчёт на сохранённых кэшах.

## Код

| Файл | Назначение |
|---|---|
| `protocol.json`, `config.py` | Закреплённые модели/revisions, девять датасетов, 25 методов и настройки |
| `data.py` | Чтение исходного ZIP, проверка источника и обработка claim |
| `features.py` | Chat-шаблон, границы claim и признаки одного примера |
| `extract.py` | GPU-batches и последовательная обработка моделей |
| `cache.py` | Атомарная запись shards и возобновление |
| `methods.py` | Исходные классы и проверка неизменности формул |
| `evaluate.py` | Scores и ROC AUC на CPU, один датасет за вызов формулы |
| `report.py` | Mean9, Average и таблицы по датасетам |

## Протокол

**Данные:** 97 488 строк из PopQA, AVeriTeC, UHEAD, Cities, Companies, CommonClaim, CounterFact, X-Fact и WildHallucinations. Метка `is_hallu=1` означает галлюцинацию. Состав строк и метки сохранены; дубликаты не удалялись. Claim обрабатывается как CRLF/CR → LF, `strip()`, первые 300 Unicode-символов; усечены 79 строк. Пути CSV, SHA, размеры, исходные cohort hashes и checkpoint revisions перечислены в [protocol.json](protocol.json).

**Вход модели:** user `Generate true statement`, assistant `<готовый claim>`, native chat template, `add_special_tokens=True`, `add_generation_prompt=False`; у Gemma `enable_thinking=False`. Готовый текст оценивается одним forward без генерации.

**Точность:** BF16 forward, FP32 log-softmax/entropy, top-512 logits в FP16; квантизации нет. Сохранены четыре поля: `sequence_logprobs`, `greedy_logprobs`, `token_entropies`, `top_k_logits`, а также token IDs и границы claim. Claim-токен на позиции `t` оценивается по logits позиции `t−1`. Формат кэша совместим с исходным `evohallu.logits_cache.v1`.

**Формулы:** все 25 локальных классов, включая Sequence Probability (`SequenceLogProb`), без изменения кода и default-параметров. Каждый метод получает весь отдельный датасет: несколько формул используют нормализацию по примерам. Это согласованное отличие от малых evaluation-батчей старого end2end. GPU microbatch не задаёт область вызова формулы.

| Отчёт | Токены | Метрика |
|---|---|---|
| `full_folded` — основной | Полный chat, включая шаблон | `max(AUC_d, 1−AUC_d)` на каждом датасете |
| `full_fixed` | Полный chat | ROC AUC с исходным направлением score |
| `claim_fixed` | Только токены claim | ROC AUC с исходным направлением score |

Больший score трактуется как большая вероятность `is_hallu=1`. **Mean9 — равновесное среднее девяти отдельных AUROC.** Folding выполняется до усреднения; он использует тестовые метки для выбора направления. В `average.md` только Mean9: 25 методов × 3 модели, жирным — максимум в каждой строке.

## Запуск из корня репозитория

Для извлечения использовались Python 3.12, PyTorch 2.8.0 / CUDA 12.8 и один L40S 48 GB. Подойдёт соответствующее готовое GPU-окружение. Установка зависимостей сама по себе GPU не арендует.

```sh
python -m pip install -r experiments/feature_transfer/requirements-gpu.txt
python -m experiments.feature_transfer.extract \
  --data-archive /path/to/datasets.zip \
  --output-dir experiments/feature_transfer/artifacts/features \
  --models llama gemma olmo \
  --batch-size 32 --shard-size 512
```

Нужен исходный `datasets.zip` с SHA из протокола и доступ к закреплённым моделям Hugging Face. Ключи передаются обычной авторизацией окружения; в код их добавлять не нужно. Повтор той же команды возобновляет только проверенные committed shards. Параллельная запись двумя CPU-потоками перекрывается с GPU forward, порядок строк восстанавливается после сортировки по длине.

Для CPU-оценки можно использовать отдельное окружение:

```sh
python -m pip install -r experiments/feature_transfer/requirements.txt
python -m experiments.feature_transfer.evaluate \
  --cache-dir experiments/feature_transfer/artifacts/features \
  --output-dir experiments/feature_transfer/artifacts/report
```

Получаются scores для всех примеров, `metrics_long.csv`, `macro9.csv`, `average.md` и три подробных отчёта. Для полного результата нужны все **27 кэшей**. `--allow-partial` разрешает CPU smoke-оценку неполного набора без публикации Mean9. Извлечение с `--limit` или `--dtype fp32` помечает кэши как pilot-only, даже при совпадении числа строк с полным датасетом. FP32-пилот следует запускать с `--batch-size 1` в отдельный output-dir.

```sh
python -m unittest discover -s tests -p test_feature_transfer.py -v
```

Проверки ядра признаков требуют PyTorch; остальные тесты и CPU-оценка работают без него.

## Результаты и ограничения

Исходный прогон завершён: 292 464 model/example пары, 27 кэшей, 1 350 model/dataset/method/view результатов и 225 Mean9. Все AUROC и средние независимо проверены sklearn. Извлечение заняло 27 мин 49 с; кэши (~28.16 GB) выгружены, RunPod удалён. Верхняя оценка расходов с пилотами и диском — $1.14. В Git включены небольшие агрегированные результаты; веса, кэши и credentials не включены.

- Full chat сохраняет область токенов доступной реализации; claim only отдельно проверяет влияние шаблона. Перенос одновременно меняет модель, токенизацию и шаблон.
- В коротком BF16/FP32-пилоте `MinimalLogProb/full` на Llama оказался чувствителен к шаблонным токенам и численным изменениям; microbatch также различался. Полного FP32-повтора нет.
- Доверительные интервалы и статистическая значимость не оценивались. Соответствие локальных классов именам Evo-GPT/Evo-Anth из статьи не установлено; отсутствующие внешние методы не добавлялись.
