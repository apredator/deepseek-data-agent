# Инструкция по использованию (USAGE)

Практическое руководство по HR/payroll-агенту: как поставить, запустить,
задавать вопросы, подключить свои данные и что делать, если что-то не работает.

> ⚠️ Все данные, которые генерирует проект, **синтетические и вымышленные**.
> Реальные данные клиентов не используются. См. `data/DISCLAIMER.txt`.

---

## 1. Что это такое

Агент аудита HR/payroll-данных SAP HCM. Он получает вопрос на естественном языке
(«есть ли ретро-расчёт за закрытый период без основания?») и **сам решает**, какие
read-only проверки вызвать, чтобы ответить. Каждый вызов инструмента виден в
трассе: имя, аргументы, сколько записей вернул, сколько занял.

Коротко о режимах:

| Режим | Что делает | Нужен API-ключ? |
|-------|-----------|-----------------|
| `--audit` | детерминированный прогон всех проверок + scorecard | нет |
| `--mock` | полный tool-calling цикл со скриптованным LLM | нет |
| обычный диалог | реальные вопросы к DeepSeek с вызовом инструментов | да |
| `api.py` | тот же агент как HTTP-сервис | да |
| `demo_app.py` | Streamlit-чат с трассой вызовов | да (или offline-режим) |

---

## 2. Установка

```bash
git clone https://github.com/apredator/deepseek-data-agent.git
cd deepseek-data-agent

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Опциональные зависимости (не обязательны):

```bash
pip install -r requirements-optional.txt   # sentence-transformers, datasets, huggingface_hub
```

## 3. Настройка

```bash
cp .env.example .env
```

Минимум, что нужно для живого диалога — ключ DeepSeek:

```dotenv
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

**Ключ не обязателен**, чтобы посмотреть демо: разделы 4.1–4.3 работают без него.
Все параметры (лимит шагов, порог скачка оклада, размер синтетики, пути, лог-уровень)
описаны в `.env.example` — хардкода в коде нет.

## 4. Запуск

### 4.1 Сгенерировать синтетический датасет

```bash
python generate_synthetic_dataset.py
```

Создаёт в `data/`:

* `pa0001_org.csv` — оргназначение (Pernr, Orgeh, Kostl, Planstelle, Begda, Endda);
* `pa0008_basic_pay.csv` — основной оклад;
* `pa0014_recurring.csv` — постоянные надбавки;
* `rt_payroll_results.csv` — результаты расчёта;
* `anomalies_manifest.json` — **ground truth**: где именно лежит каждая внедрённая аномалия;
* `DISCLAIMER.txt` — пометка о синтетичности данных.

Полезные флаги:

```bash
python generate_synthetic_dataset.py --employees 1000 --periods 12 --seed 7
python generate_synthetic_dataset.py --no-injection-probe    # без payload для security-теста
```

### 4.2 Детерминированный аудит (без LLM)

```bash
python main.py hr --audit
```

Выводит таблицу «найдено / ожидалось», метрику detection rate и false positives.

### 4.3 Оффлайн-демо tool-calling цикла (без API-ключа)

```bash
python main.py hr --mock
python main.py hr --mock -q "Сделай полный аудит payroll-данных"
```

Скриптованный LLM вызывает все шесть проверок, и вы видите настоящую трассу
вызовов и настоящие результаты инструментов.

### 4.4 Интерактивный диалог (нужен ключ)

```bash
python main.py hr
python main.py hr -q "Есть ли ретро-расчёт за закрытый период без основания?"
python main.py hr -q "Вопрос 1" -q "Вопрос 2" --data data
```

Команды в интерактивном режиме:

| Команда | Действие |
|---------|----------|
| `<вопрос>` | задать вопрос агенту |
| `audit` | детерминированный аудит всех проверок + scorecard |
| `tools` | список read-only инструментов |
| `security` | скан датасета на prompt injection |
| `trace on` / `trace off` | показывать/скрывать трассу вызовов |
| `clear` | очистить состояние агента |
| `help` | справка |
| `quit` | выход |

Другие флаги CLI: `--tools`, `--security`, `--provider deepseek|hf|mock`,
`--no-trace`, `--regenerate`.

### 4.5 HTTP-сервис (FastAPI)

```bash
uvicorn api:app --reload --port 8000
```

```bash
# проверка живости
curl -s localhost:8000/health | jq

# каталог инструментов
curl -s localhost:8000/tools | jq

# вопрос агенту: ответ + трасса вызовов
curl -s localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"question": "Проверь на дублирующиеся начисления по одному Lohnart"}' | jq

# детерминированный аудит без LLM
curl -s -X POST localhost:8000/audit | jq
```

Ответ `/analyze`:

```json
{
  "answer": "Найдено 5 дублирующихся начислений Lohnart 2000 в периоде 2024P10 …",
  "mode": "tool_calling",
  "iterations": 2,
  "used_tools": ["check_duplicate_wage_types"],
  "trace": [
    {"iteration": 1, "name": "check_duplicate_wage_types",
     "arguments": {}, "ok": true, "count": 5, "duration_ms": 7.1}
  ],
  "guardrails": {"read_only": true, "violations": [], "injection_patterns": []}
}
```

Прочие эндпоинты: `GET /datasets`, `POST /datasets/load`,
`POST /analyze/stream-trace`.

### 4.6 Streamlit-демо

```bash
streamlit run demo_app.py
```

Открывается браузер: выбор/генерация датасета, чат, трасса вызовов под каждым
ответом, частота вызовов инструментов, кнопка «Детерминированный аудит» и
«Проверить данные на prompt injection». Если ключа нет — включите тумблер
**Offline demo mode** в сайдбаре.

### 4.7 Готовые примеры

```bash
python example_usage.py 5   # датасет + scorecard                (без ключа)
python example_usage.py 6   # tool-calling цикл на mock-LLM       (без ключа)
python example_usage.py 7   # security-тест prompt injection      (без ключа)
python example_usage.py 8   # живые вопросы к агенту              (нужен ключ)
```

---

## 5. Примеры вопросов агенту

* «Есть ли сотрудники с ретро-расчётом за закрытый период без основания?»
* «Покажи всех, у кого оклад вырос более чем на 40 % без записи о повышении в оргназначении.»
* «Проверь на дублирующиеся начисления по одному Lohnart в одном периоде.»
* «Сведи расхождение между Kostl (МВЗ) в оргназначении и в результатах расчёта.»
* «Есть ли начисления после даты увольнения?»
* «Покажи периоды с отрицательным net pay.»
* «Сделай сводку gross pay по орг-единицам за 2024P10.»
* «Покажи полную карточку сотрудника 10000042.» ← агент вызовет
  `get_employee_record` по Pernr, который до этого нашла проверка.

Агент отвечает на языке вопроса и обязан ссылаться на конкретные Pernr, периоды
и суммы — если он не вызвал ни одного инструмента, доверять ответу не стоит.

---

## 6. Как подключить свои данные

Агент читает четыре CSV с фиксированными именами из каталога `--data` (по умолчанию
`data/`). Достаточно подготовить свои выгрузки с теми же колонками:

| Файл | Обязательные колонки |
|------|----------------------|
| `pa0001_org.csv` | `Pernr, Orgeh, Kostl, Planstelle, Begda, Endda` (+ опц. `Massn, Position, Nachn, Vorna`) |
| `pa0008_basic_pay.csv` | `Pernr, Lohnart, Betrag, Waehrung, Begda, Endda` |
| `pa0014_recurring.csv` | `Pernr, Lohnart, Betrag, Periodizitaet` |
| `rt_payroll_results.csv` | `Pernr, Abrechnungsperiode, Lohnart, Betrag, InPeriod, ForPeriod, Retro, Kostl` |

Важные соглашения:

* `Pernr` и `Lohnart` — **строки** (`Lohnart = "1000"`, не `1000`). Агент сам
  приводит их к строке при загрузке, но в исходнике лучше хранить как текст.
* Период — `YYYYPMM` (`2024P07`), но инструменты принимают и `2024-07`, `202407`, `7`.
* Даты — `YYYY-MM-DD`; «бессрочная» валидность — `2199-12-31`
  (`datetime64[ns]` в pandas не умеет 9999-12-31).
* Порог скачка оклада — `SALARY_JUMP_THRESHOLD_PCT`; закрытые периоды —
  `SYNTHETIC_CLOSED_PERIODS` (для своих данных их можно положить в
  `anomalies_manifest.json` → `meta.closed_periods`).
* `anomalies_manifest.json` для своих данных не обязателен: без него работают все
  проверки, но пропадёт scorecard «найдено / ожидалось».

Запуск на своих данных:

```bash
python main.py hr --data /path/to/export --audit
python main.py hr --data /path/to/export -q "Проверь дубли Lohnart"
```

---

## 7. Конфигурация (`.env`)

| Переменная | Назначение | По умолчанию |
|-----------|-----------|--------------|
| `DEEPSEEK_API_KEY` | ключ DeepSeek | — |
| `DEEPSEEK_MODEL` | модель | `deepseek-chat` |
| `DEEPSEEK_API_BASE` | базовый URL API | `https://api.deepseek.com/v1` |
| `LLM_PROVIDER` | `deepseek` / `hf` / `mock` | `deepseek` |
| `HF_TOKEN`, `HF_MODEL`, `ENABLE_HF_FALLBACK` | резервный бэкенд Hugging Face | выключен |
| `MAX_TOOL_ITERATIONS` | жёсткий лимит шагов цикла | `5` |
| `ALLOW_REACT_FALLBACK` | текстовый ReAct, если нет function calling | `True` |
| `AGENT_READ_ONLY` | запрет изменяющих инструментов | `True` |
| `ENABLE_SEMANTIC_TOOLS` | добавить семантический поиск похожих ФИО | `False` |
| `SALARY_JUMP_THRESHOLD_PCT` | порог скачка оклада | `40` |
| `DATA_DIR` / `OUTPUT_DIR` / `LOG_DIR` | каталоги | `data` / `outputs` / `logs` |
| `LOG_LEVEL`, `DEBUG` | логирование | `INFO`, `False` |
| `SYNTHETIC_NUM_EMPLOYEES`, `SYNTHETIC_NUM_PERIODS`, `SYNTHETIC_CLOSED_PERIODS`, `SYNTHETIC_SEED` | параметры генератора | `600`, `12`, `9`, `42` |
| `SYNTHETIC_INJECTION_PROBE` | внедрять payload для security-теста | `True` |
| `REQUEST_TIMEOUT_SECONDS` | таймаут запроса к LLM | `60` |

---

## 8. Тесты

```bash
pytest -q            # 106 тестов
pytest -q -k security
pytest -q tests/test_hr_tools.py
```

Покрытие: по одному тесту на каждый инструмент (сверка с ground truth),
детерминизм генератора, round-trip save/load, guardrails, security-тест на
prompt injection, цикл агента (несколько вызовов, заблокированный инструмент,
лимит шагов, ReAct-fallback, падение LLM), слой LLM (парсинг tool-calls,
фолбэк при отсутствии function calling), эндпоинты FastAPI и headless-тест
Streamlit-интерфейса через `AppTest`.
Тесты API пропускаются автоматически, если не установлены `fastapi`/`httpx`
(`pip install -r requirements.txt` их ставит).
CI: `.github/workflows/tests.yml` (GitHub Actions, Python 3.10 и 3.12).

---

## 9. Если что-то не работает

**`DEEPSEEK_API_KEY is not set`**
Ключа нет в `.env`. Либо добавьте его, либо используйте `--audit`, `--mock`
или offline-режим в Streamlit — они работают без ключа.

**`Dataset directory ... is incomplete, missing: ...`**
Каталог `data/` неполный. Запустите `python generate_synthetic_dataset.py`.

**Агент пишет «Tool-call budget exhausted»**
Модель не уложилась в `MAX_TOOL_ITERATIONS`. Задайте более узкий вопрос
(конкретный тип аномалии, период, Pernr) или увеличьте лимит в `.env`.

**`Tool 'delete_all_records' is not in the allow-list`**
Это guardrails: агент строго read-only. Так и должно быть. Используйте только
инструменты из `python main.py hr --tools`.

**`GuardrailViolation: Tool-calling loop limit reached`**
Защита от зацикливания. Сформулируйте вопрос точнее или поднимите лимит.

**Модель отвечает без вызова инструментов**
Проверьте `LLM_PROVIDER` и модель. Если у модели нет function calling, агент
автоматически перейдёт на ReAct (`mode: react_fallback`) при
`ALLOW_REACT_FALLBACK=True`.

**Проверки ничего не находят на моих данных**
Убедитесь, что `Lohnart` — строки вида `"1000"`/`"2000"`, период в формате
`YYYYPMM`, а «бессрочная» дата — `2199-12-31`, а не `9999-12-31`.

**`prompt-injection payload в данных`**
Это нормально для демо-датасета: генератор намеренно кладёт payload в одно ФИО.
Агент обрабатывает его как данные. Отключить: `--no-injection-probe` при генерации
или `SYNTHETIC_INJECTION_PROBE=False`.

**Мало данных / хочу больше**
`SYNTHETIC_NUM_EMPLOYEES=2000 python generate_synthetic_dataset.py --employees 2000`

---

## 10. Чек-лист приёмки (Definition of Done)

- [x] Агент находит **все 6 типов** внедрённых аномалий (31/31) через
      естественноязыковые вопросы, без хардкода ответа.
- [x] Полный tool-calling цикл работает и логируется (трасса в CLI/API/UI).
- [x] Prompt injection тест проходит: данные из ячеек не становятся инструкциями.
- [x] CI зелёный, тесты покрывают все функции `hr_tools.py`.
- [x] `README.md` (EN) и `CASE_STUDY.md` готовы к показу заказчику.
- [ ] Демо-видео 2–3 минуты и скриншоты Streamlit — снять вручную
      (сценарий показа — раздел 5 `CASE_STUDY.md`).

## 11. Куда смотреть дальше

* `README.md` — архитектура, результаты, security-раздел.
* `CASE_STUDY.md` — бизнес-контекст и метрики для заказчика.
* `data/anomalies_manifest.json` — ground truth для проверки ответов агента.
* `python main.py hr --tools` — актуальный список инструментов.
