import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "article_generation.log"
ARTICLE_FILE = PROJECT_ROOT / "article.md"
DOTENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_MODEL = "gpt-5.4-nano"


logger = logging.getLogger(__name__)


class ArticleGenerationError(Exception):
    """User-friendly error for expected validation and configuration issues."""


def configure_logging() -> None:
    """Send logs both to the console and to article_generation.log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
        force=True,
    )


def load_environment() -> bool:
    """Load environment variables from .env if the file exists."""
    if DOTENV_FILE.exists():
        return load_dotenv(DOTENV_FILE)
    return False


def parse_arguments() -> str:
    """Read the article topic from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate a science-popular article with a four-step LangChain pipeline."
    )
    parser.add_argument(
        "topic",
        nargs="*",
        help="Topic or short description of the science-popular article.",
    )
    args = parser.parse_args()
    topic = " ".join(args.topic).strip()
    if not topic:
        raise ArticleGenerationError(
            'Не передана тема статьи. Пример: python script.py "Почему люди видят сны"'
        )
    return topic


def build_llm() -> ChatOpenAI:
    """Create an OpenAI-compatible chat model from environment settings."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ArticleGenerationError(
            "Переменная OPENAI_API_KEY не задана. Укажите её в окружении или в .env."
        )

    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    logger.info("Создание LLM: model=%s base_url=%s", model_name, base_url or "default")

    # Pass the key explicitly so the script works even if the runtime reads env vars loosely.
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 1.0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def parse_json_response(text: str, stage_name: str) -> dict[str, Any]:
    """Parse a JSON object from the model output and provide a clear error message."""
    cleaned = text.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1).strip())
            except json.JSONDecodeError as exc:
                raise ArticleGenerationError(
                    f"{stage_name}: модель вернула некорректный JSON."
                ) from exc
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise ArticleGenerationError(
                        f"{stage_name}: модель вернула некорректный JSON."
                    ) from exc
            else:
                raise ArticleGenerationError(
                    f"{stage_name}: модель вернула текст, который нельзя разобрать как JSON."
                )

    if not isinstance(parsed, dict):
        raise ArticleGenerationError(f"{stage_name}: ожидается JSON-объект.")

    return parsed


def validate_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the analysis payload returned by analysis_chain."""
    required_fields = [
        "topic",
        "main_problem",
        "target_audience",
        "article_goal",
        "key_questions",
        "tone",
        "complexity_level",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ArticleGenerationError(
            "analysis_chain: отсутствуют обязательные поля: " + ", ".join(missing)
        )

    if not isinstance(data["key_questions"], list) or len(data["key_questions"]) < 2:
        raise ArticleGenerationError(
            "analysis_chain: key_questions должен быть списком из нескольких вопросов."
        )

    return data


def validate_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the plan payload returned by plan_chain."""
    required_fields = [
        "title",
        "subtitle",
        "sections",
        "intro_idea",
        "conclusion_idea",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ArticleGenerationError(
            "plan_chain: отсутствуют обязательные поля: " + ", ".join(missing)
        )

    sections = data["sections"]
    if not isinstance(sections, list):
        raise ArticleGenerationError("plan_chain: sections должен быть списком.")
    if not 4 <= len(sections) <= 7:
        raise ArticleGenerationError("plan_chain: sections должен содержать 4-7 разделов.")

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            raise ArticleGenerationError(
                f"plan_chain: раздел #{index} должен быть JSON-объектом."
            )
        section_required = ["heading", "purpose", "key_points"]
        section_missing = [field for field in section_required if field not in section]
        if section_missing:
            raise ArticleGenerationError(
                f"plan_chain: в разделе #{index} не хватает полей: "
                + ", ".join(section_missing)
            )
        if not isinstance(section["key_points"], list) or not section["key_points"]:
            raise ArticleGenerationError(
                f"plan_chain: key_points в разделе #{index} должен быть непустым списком."
            )

    return data


def build_analysis_chain(llm: ChatOpenAI):
    """Create the first chain: topic -> structured analysis."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent(
                    """
                    Ты профессиональный копирайтер, помогаешь готовить научно-популярные статьи.
                    Проанализируй тему и верни только валидный JSON без пояснений и без markdown.
                    """
                ).strip(),
            ),
            (
                "human",
                dedent(
                    """
                    Тема статьи:
                    {topic}

                    Сформируй JSON строго по схеме:
                    {{
                      "topic": "...",
                      "main_problem": "...",
                      "target_audience": "...",
                      "article_goal": "...",
                      "key_questions": ["...", "..."],
                      "tone": "...",
                      "complexity_level": "..."
                    }}

                    Требования:
                    - определи главную тему;
                    - определи основную проблему или интригу статьи;
                    - определи целевую аудиторию;
                    - сформулируй цель статьи;
                    - укажи 2-5 ключевых вопросов;
                    - задай подходящий тон;
                    - укажи уровень сложности объяснения.
                    """
                ).strip(),
            ),
        ]
    )

    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda text: validate_analysis(parse_json_response(text, "analysis_chain"))
    )


def build_plan_chain(llm: ChatOpenAI):
    """Create the second chain: analysis -> article plan."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent(
                    """
                    Ты профессиональный редактор с широким кругозором, создаёшь подробный и логичный план научно-популярной статьи.
                    Верни только валидный JSON без пояснений и без markdown.
                    """
                ).strip(),
            ),
            (
                "human",
                dedent(
                    """
                    Используй результат анализа:
                    {analysis_json}

                    Сформируй JSON строго по схеме:
                    {{
                      "title": "...",
                      "subtitle": "...",
                      "sections": [
                        {{
                          "heading": "...",
                          "purpose": "...",
                          "key_points": ["...", "..."]
                        }}
                      ],
                      "intro_idea": "...",
                      "conclusion_idea": "..."
                    }}

                    Требования:
                    - придумай цепляющий заголовок;
                    - добавь подзаголовок;
                    - создай логичную структуру статьи;
                    - сделай 4-7 разделов;
                    - для каждого раздела укажи цель и ключевые тезисы;
                    - продумай идею введения;
                    - продумай идею заключения.
                    """
                ).strip(),
            ),
        ]
    )

    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda text: validate_plan(parse_json_response(text, "plan_chain"))
    )


def build_draft_chain(llm: ChatOpenAI):
    """Create the third chain: analysis + plan -> article draft."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent(
                    """
                    Ты редактор-профи, пишешь полный черновик научно-популярной статьи.
                    Пиши ясно, живо и без академической перегрузки.
                    Верни только Markdown-текст без пояснений.
                    """
                ).strip(),
            ),
            (
                "human",
                dedent(
                    """
                    Используй анализ и план ниже.

                    Анализ:
                    {analysis_json}

                    План:
                    {plan_json}

                    Напиши черновик статьи на русском языке.
                    Требования:
                    - 600-1000 слов;
                    - короткие абзацы;
                    - понятные объяснения простыми словами;
                    - используй аналогии и примеры;
                    - не выдумывай точные факты, даты, исследования или статистику;
                    - если точные данные неизвестны, формулируй осторожно;
                    - структура Markdown:
                      # Заголовок
                      вводный абзац
                      ## Разделы
                      заключение.
                    """
                ).strip(),
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def build_review_chain(llm: ChatOpenAI):
    """Create the fourth chain: draft -> edited final article."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent(
                    """
                    Ты редактор научно-популярных текстов.
                    Проверь структуру, плавность переходов, повторы и тон.
                    Верни только итоговый Markdown-текст без комментариев.
                    """
                ).strip(),
            ),
            (
                "human",
                dedent(
                    """
                    Отредактируй черновик статьи:

                    {draft}

                    Требования:
                    - сохрани Markdown-разметку;
                    - сделай текст более человечным и связным;
                    - убери повторы и пустые фразы;
                    - не добавляй непроверенные факты;
                    - не сокращай статью слишком сильно;
                    - сохрани научно-популярный стиль.
                    """
                ).strip(),
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def run_stage(stage_name: str, chain: Any, payload: dict[str, Any] | str) -> Any:
    """Log the start/end of each chain and execute it."""
    logger.info("Starting %s", stage_name)
    result = chain.invoke(payload)
    logger.info("Completed %s", stage_name)
    return result


def save_article(article_text: str) -> None:
    """Persist the final article to article.md."""
    ARTICLE_FILE.write_text(article_text.strip() + "\n", encoding="utf-8")
    logger.info("Сохранён файл %s", ARTICLE_FILE.name)


def main() -> int:
    configure_logging()
    logger.info("Starting program")

    try:
        env_loaded = load_environment()
        logger.info(".env loaded: %s", "yes" if env_loaded else "no")

        topic = parse_arguments()
        logger.info("Получена тема статьи: %s", topic)

        llm = build_llm()

        analysis_chain = build_analysis_chain(llm)
        plan_chain = build_plan_chain(llm)
        draft_chain = build_draft_chain(llm)
        review_chain = build_review_chain(llm)

        analysis = run_stage("analysis_chain", analysis_chain, {"topic": topic})
        plan = run_stage(
            "plan_chain",
            plan_chain,
            {"analysis_json": json.dumps(analysis, ensure_ascii=False, indent=2)},
        )
        draft = run_stage(
            "draft_chain",
            draft_chain,
            {
                "analysis_json": json.dumps(analysis, ensure_ascii=False, indent=2),
                "plan_json": json.dumps(plan, ensure_ascii=False, indent=2),
            },
        )
        final_article = run_stage("review_chain", review_chain, {"draft": draft})

        if not isinstance(final_article, str) or not final_article.strip():
            raise ArticleGenerationError("review_chain: итоговая статья оказалась пустой.")

        save_article(final_article)
        logger.info("Program completed successfully")
        print("Статья создана: article.md")
        return 0

    except ArticleGenerationError as exc:
        logger.error(str(exc))
        print(f"Ошибка: {exc}")
        return 1
    except Exception:
        logger.exception("Unexpected exception")
        print("Произошла неожиданная ошибка. Подробности записаны в article_generation.log")
        return 1


if __name__ == "__main__":
    sys.exit(main())
