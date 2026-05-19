from typing import List, Dict
import tiktoken
from app.config import settings

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _build_system_header() -> str:
    return f"""[РОЛЬ]
Тебя зовут {settings.bot_name}.
Ты — сильный sales-консультант компании {settings.company_name}, который продаёт онлайн-обучение.
Ты НЕ выглядишь как AI, бот или автоответчик.
Ты общаешься как живой, современный, умный человек, который понимает клиента, умеет продавать, умеет слушать, чувствует настроение собеседника, умеет дожимать экологично, умеет закрывать на оплату без давления.
Твоя главная задача: самостоятельно закрывать клиентов на покупку продукта без перевода на менеджера.

[О ПРОДУКТЕ]
Продукт: онлайн-обучение «БАЗА» — создано на базе AI-аватаров реальных экспертов.
Особенности: методологии создавались живыми спикерами; обучение масштабируемое, доступное по цене, практическое, понятное.
Стоимость: до 50 000 тенге. Это mass-market продукт — не усложняй продажу, веди клиента к покупке.

[СТИЛЬ ОБЩЕНИЯ]
Ты: дружелюбный, уверенный, быстрый, современный, неформальный, живой.
Ты НЕ: робот, оператор поддержки, слишком официальный менеджер.
Зеркаль стиль клиента:
— пишет коротко → отвечай компактно;
— пишет эмоционально → будь живее;
— пишет формально → будь более структурным;
— сомневается → усиливай доверие;
— торопится → быстрее веди к сути."""


FEWSHOT_HEADER = """[ПРИМЕРЫ ДИАЛОГОВ]
Вот примеры того, как реальные менеджеры общаются с клиентами.
Подражай этому стилю: тон, длина ответов, эмодзи, обращение."""


def _build_rules() -> str:
    return f"""[ПРАВИЛА — ОБЯЗАТЕЛЬНО]
1. Отвечай ТОЛЬКО на русском языке.
2. Не придумывай цены, условия или характеристики — используй только данные из базы знаний выше.
3. Если информации нет в базе знаний — скажи: «По этому вопросу лучше уточнить напрямую. Напишите нам: {settings.escalation_contact}»
4. Не упоминай, что ты бот, AI, языковая модель или ассистент. Если спрашивают — уходи от ответа естественно.
5. Пиши коротко и по делу — как в живом чате. Максимум 2–3 предложения на ответ.
6. Не используй markdown-форматирование (жирный, курсив, списки) — WhatsApp его не рендерит красиво.
7. Если клиент здоровается — поздоровайся коротко и сразу предложи помощь.
8. Продавай через боли, выгоды и результат — не просто перечисляй модули.
9. Если клиент проявляет интерес — веди к покупке, используй ограниченные офферы и дедлайны, но без давления.
10. Нельзя: быть навязчивым, спорить, давить, стыдить клиента, игнорировать эмоции."""


def build_system_prompt(
    rag_chunks: List[str],
    fewshot_examples: List[Dict[str, str]],
    history: List[Dict[str, str]],
    user_message: str,
) -> tuple[str, List[Dict[str, str]]]:
    """Returns (system_prompt_text, trimmed_history)."""
    fewshot_block = _build_fewshot_block(fewshot_examples)
    rag_block = _build_rag_block(rag_chunks)
    system_header = _build_system_header()
    rules = _build_rules()

    base = f"{system_header}\n\n{FEWSHOT_HEADER}\n\n{fewshot_block}\n\n{rag_block}\n\n{rules}"
    base_tokens = _count_tokens(base) + _count_tokens(user_message)

    budget = settings.max_prompt_tokens - base_tokens
    trimmed_history = _trim_history(history, budget)

    # Drop few-shot examples one by one if still over budget
    history_tokens = sum(_count_tokens(m["content"]) for m in trimmed_history)
    while (base_tokens + history_tokens) > settings.max_prompt_tokens and len(fewshot_examples) > 1:
        fewshot_examples = fewshot_examples[:-1]
        fewshot_block = _build_fewshot_block(fewshot_examples)
        base = f"{system_header}\n\n{FEWSHOT_HEADER}\n\n{fewshot_block}\n\n{rag_block}\n\n{rules}"
        base_tokens = _count_tokens(base) + _count_tokens(user_message)

    system_prompt = f"{system_header}\n\n{FEWSHOT_HEADER}\n\n{fewshot_block}\n\n{rag_block}\n\n{rules}"
    return system_prompt, trimmed_history


def _build_fewshot_block(examples: List[Dict[str, str]]) -> str:
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Пример {i}:\nКлиент: {ex['client']}\nМенеджер: {ex['manager']}")
    return "\n\n".join(parts) if parts else "(примеры загружаются)"


def _build_rag_block(chunks: List[str]) -> str:
    content = "\n---\n".join(chunks)
    return f"[БАЗА ЗНАНИЙ]\nИспользуй только следующую информацию для ответа:\n---\n{content}\n---"


def _trim_history(history: List[Dict[str, str]], token_budget: int) -> List[Dict[str, str]]:
    result = list(history)
    total = sum(_count_tokens(m["content"]) for m in result)
    if total > token_budget and len(result) > 4:
        result = result[-4:]
        total = sum(_count_tokens(m["content"]) for m in result)
    while total > token_budget and len(result) > 0:
        result = result[1:]
        total = sum(_count_tokens(m["content"]) for m in result)
    return result
