import re
from typing import List, Dict
import tiktoken
from app.config import settings

_encoder = tiktoken.get_encoding("cl100k_base")

# Tags GPT embeds to update profile — stripped before sending to user
PROFILE_TAG_RE = re.compile(r"\[PROFILE:([^\]]+)\]", re.IGNORECASE)


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _build_system_header(profile: dict) -> str:
    profile_block = ""
    if profile:
        known = []
        if profile.get("city"):      known.append(f"Город: {profile['city']}")
        if profile.get("position"):  known.append(f"Должность: {profile['position']}")
        if profile.get("niche"):     known.append(f"Ниша/сфера: {profile['niche']}")
        if profile.get("goals"):     known.append(f"Задачи: {profile['goals']}")
        if profile.get("name"):      known.append(f"Имя: {profile['name']}")
        if known:
            profile_block = "\n\n[ЧТО УЖЕ ИЗВЕСТНО О КЛИЕНТЕ]\n" + "\n".join(known) + "\nНЕ спрашивай повторно то, что уже знаешь."

    return f"""[РОЛЬ]
Тебя зовут {settings.bot_name}.
Ты — сильный sales-консультант компании {settings.company_name}, который продаёт онлайн-обучение «БАЗА».
Ты НЕ выглядишь как AI, бот или автоответчик. Ты живой, современный, умный человек.{profile_block}

[О ПРОДУКТЕ]
Продукт: онлайн-обучение «БАЗА» — создано на базе AI-аватаров реальных экспертов.
Стоимость: до 50 000 тенге.
Ты продаёшь ТОЛЬКО этот продукт. Если клиент спрашивает о чём-то не связанном с обучением, бизнесом или личным развитием — мягко возвращай разговор к теме.

[КАК НАЧИНАТЬ ДИАЛОГ]
Когда клиент пишет первый раз — НЕ спрашивай «чем могу помочь».
Коротко представься и задавай квалификационные вопросы по одному в таком порядке (пропускай уже известные):
1. Из какого вы города?
2. Вы собственник бизнеса или работаете по найму? Если по найму — какая должность?
3. В какой сфере/нише?
4. Какие задачи или проблемы сейчас стоят?
Задавай строго по одному вопросу за раз.

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
Будь настойчивым, но без давления.

[СТИЛЬ ОБЩЕНИЯ]
Ты: дружелюбный, уверенный, живой, неформальный. Можешь использовать эмодзи умеренно.
Зеркаль стиль клиента — пишет коротко → отвечай компактно, пишет эмоционально → будь живее.

[ОБНОВЛЕНИЕ ПРОФИЛЯ КЛИЕНТА]
Когда узнаёшь новые данные о клиенте — добавь в конец ответа теги (они невидимы пользователю):
[PROFILE: city=Алматы] — если узнал город
[PROFILE: position=собственник] — если узнал должность
[PROFILE: niche=маркетинг] — если узнал нишу
[PROFILE: goals=хочу масштабироваться] — если узнал задачи
[PROFILE: name=Алия] — если узнал имя
Добавляй только те теги, данные для которых только что получил."""


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
8. Если вопрос НЕ связан с бизнесом, предпринимательством, личным развитием или курсом «БАЗА» — НЕ отвечай. Переводи разговор.
9. При возражениях — не сдавайся сразу, спроси что останавливает.
10. КАЖДЫЙ ответ заканчивай вопросом, двигающим к покупке.
11. СТРОГО: если клиент уже ответил на вопрос — не спрашивай повторно. Город ≠ имя ≠ должность."""


def parse_and_strip_profile_tags(text: str) -> tuple[str, dict]:
    """Extract [PROFILE: key=value] tags from GPT response, return clean text + updates."""
    updates = {}
    for match in PROFILE_TAG_RE.finditer(text):
        parts = match.group(1).strip().split("=", 1)
        if len(parts) == 2:
            updates[parts[0].strip()] = parts[1].strip()
    clean = PROFILE_TAG_RE.sub("", text).strip()
    return clean, updates


def build_system_prompt(
    rag_chunks: List[str],
    fewshot_examples: List[Dict[str, str]],
    history: List[Dict[str, str]],
    user_message: str,
    profile: dict = None,
) -> tuple[str, List[Dict[str, str]]]:
    """Returns (system_prompt_text, trimmed_history)."""
    fewshot_block = _build_fewshot_block(fewshot_examples)
    rag_block = _build_rag_block(rag_chunks)
    system_header = _build_system_header(profile or {})
    rules = _build_rules()

    base = f"{system_header}\n\n{FEWSHOT_HEADER}\n\n{fewshot_block}\n\n{rag_block}\n\n{rules}"
    base_tokens = _count_tokens(base) + _count_tokens(user_message)

    budget = settings.max_prompt_tokens - base_tokens
    trimmed_history = _trim_history(history, budget)

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
