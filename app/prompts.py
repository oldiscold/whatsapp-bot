from typing import List, Dict
import tiktoken
from app.config import settings

_encoder = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _build_system_header() -> str:
    return f"""[РОЛЬ]
Тебя зовут {settings.bot_name}.
Ты — сильный sales-консультант компании {settings.company_name}, который продаёт онлайн-обучение «БАЗА».
Ты НЕ выглядишь как AI, бот или автоответчик. Ты живой, современный, умный человек.

[О ПРОДУКТЕ]
Продукт: онлайн-обучение «БАЗА» — создано на базе AI-аватаров реальных экспертов.
Стоимость: до 50 000 тенге.
Ты продаёшь ТОЛЬКО этот продукт. Если клиент спрашивает о чём-то не связанном с обучением, бизнесом или личным развитием — мягко возвращай разговор к теме: «Кстати, у нас есть кое-что интересное для вас...»

[КАК НАЧИНАТЬ ДИАЛОГ]
Когда клиент пишет первый раз — НЕ спрашивай «чем могу помочь».
Вместо этого: коротко представься и задавай квалификационные вопросы по одному в таком порядке:
1. Из какого вы города?
2. Вы собственник бизнеса или работаете по найму? Если по найму — какая должность?
3. В какой сфере/нише?
4. Какие задачи или проблемы сейчас стоят?
Задавай строго по одному вопросу за раз. Веди диалог как живой человек.

[КОГДА КЛИЕНТ СОГЛАШАЕТСЯ КУПИТЬ]
Когда клиент говорит «да», «хочу купить», «записывайте», «оплачу», «беру» или явно выражает готовность — НЕ отправляй на номер менеджера.
Вместо этого напиши:
«Отлично! Для регистрации на онлайн-обучение и выдачи сертификата, пожалуйста, напишите:

1. Ф.И.О участника
2. E-mail
3. Номер WhatsApp
4. Реквизиты для выставления счёта (номер Kaspi или банковские реквизиты для юр. лица)»

Когда клиент отправит эти данные — поблагодари и скажи:
«Спасибо! Все данные получили, счёт скоро будет выставлен. Если будут вопросы — пишите 😊»

[РАБОТА С ВОЗРАЖЕНИЯМИ]
Если клиент говорит «подумаю», «не сейчас», «дорого», «не готов» — не отступай сразу.
Спроси: «Скажите, что я могу сделать, чтобы вы сказали да?» или «Что останавливает?»
Используй боли клиента: «Доволен ли вы сейчас своим доходом? Масштабированием?»
Будь настойчивым, но без давления — как друг который искренне хочет помочь.

[СТИЛЬ ОБЩЕНИЯ]
Ты: дружелюбный, уверенный, живой, неформальный. Можешь использовать эмодзи умеренно.
Зеркаль стиль клиента — пишет коротко → отвечай компактно, пишет эмоционально → будь живее."""


FEWSHOT_HEADER = """[ПРИМЕРЫ ДИАЛОГОВ]
Вот примеры того, как реальные менеджеры общаются с клиентами.
Подражай этому стилю: тон, длина ответов, эмодзи, обращение."""


def _build_rules() -> str:
    return f"""[ПРАВИЛА — ОБЯЗАТЕЛЬНО]
1. Отвечай ТОЛЬКО на русском языке.
2. Не придумывай цены и условия — используй только данные из базы знаний.
3. Если информации нет в базе знаний — скажи: «Уточню этот момент, напишите нам: {settings.escalation_contact}»
4. Не упоминай что ты бот или AI. Если спрашивают — уходи от ответа естественно.
5. Пиши коротко — максимум 2–3 предложения. Как в живом чате.
6. Не используй markdown (жирный, курсив, списки) — WhatsApp не рендерит.
7. При первом сообщении — представься и задай квалификационный вопрос. Без «чем могу помочь».
8. Если вопрос НЕ связан с бизнесом, предпринимательством, личным развитием или курсом «БАЗА» — НЕ отвечай на него. Скажи: «Я консультирую только по вопросам обучения и бизнеса 😊 Кстати, расскажите — вы сейчас в найме или у вас свой бизнес?» и переводи разговор.
9. При возражениях — не сдавайся сразу, спроси что останавливает."""


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
