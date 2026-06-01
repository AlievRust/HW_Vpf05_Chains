# Science Article Chain

Небольшой Python-проект, который с помощью LangChain генерирует научно-популярную статью по теме, указанной в командной строке.

Главный файл проекта: `script.py`.

## Что делает `script.py`

- принимает тему или краткое описание статьи из командной строки;
- загружает переменные окружения из `.env`, если файл существует;
- проверяет наличие `OPENAI_API_KEY`;
- создаёт OpenAI-compatible LLM через LangChain;
- последовательно запускает 4 цепочки:
  - `analysis_chain`;
  - `plan_chain`;
  - `draft_chain`;
  - `review_chain`;
- сохраняет финальную статью в `article_0001.md`, `article_0002.md` и далее;
- пишет журнал работы в консоль и в `article_generation.log`.

## Структура цепочки

### `analysis_chain`

Анализирует тему и возвращает JSON со следующими полями:

- `topic`;
- `main_problem`;
- `target_audience`;
- `article_goal`;
- `key_questions`;
- `tone`;
- `complexity_level`.

### `plan_chain`

На основе анализа строит план статьи и возвращает JSON:

- `title`;
- `subtitle`;
- `sections`;
- `intro_idea`;
- `conclusion_idea`.

### `draft_chain`

Пишет полный черновик статьи в Markdown на русском языке.

### `review_chain`

Редактирует черновик, делает текст плавнее, чище и более человечным, сохраняя Markdown-разметку.

## Создаваемые файлы

- `article_0001.md`, `article_0002.md` и далее - финальные статьи;
- `article_generation.log` - журнал работы программы.

## Установка зависимостей

Рекомендуется использовать Python 3.11+ и виртуальное окружение.

```bash
pip install -r requirements.txt
```

## Создание `.env`

Скопируйте `.env.example` в `.env` и заполните значения.

```bash
copy .env.example .env
```

## Переменные окружения

- `OPENAI_API_KEY` - API-ключ;
- `OPENAI_BASE_URL` - необязательный base URL для OpenAI-compatible API;
- `OPENAI_MODEL` - имя модели, по умолчанию `gpt-5.4-nano`.

## Пример запуска

```bash
python script.py "Почему люди видят сны"
```

После успешного выполнения в консоль будет выведено:

```text
Статья создана: article_0001.md
```

## Кратко о логировании

Программа использует стандартный модуль `logging` и пишет сообщения:

- в консоль;
- в `article_generation.log`.

Формат логов включает дату, время, уровень и сообщение.

## Примечание по структуре проекта

Проект состоит из:

- `script.py` - основной исполняемый скрипт;
- `README.md` - инструкция по запуску;
- `.env.example` - шаблон переменных окружения;
- `requirements.txt` - список зависимостей.
