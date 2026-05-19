from app.config import settings

TRIGGER_KEYWORDS = [
    "менеджер", "человек", "оператор", "руководитель",
    "жалоба", "претензия", "возврат", "договор", "скидка",
    "индивидуальные условия", "не работает", "обман",
]

ESCALATION_TEMPLATE = (
    "По этому вопросу лучше свяжитесь с нами напрямую.\n"
    "Напишите нам: {contact}"
)


def build_escalation_reply() -> str:
    return ESCALATION_TEMPLATE.format(contact=settings.escalation_contact)


def check_keywords(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in TRIGGER_KEYWORDS)


def check_rag_score(best_score: float) -> bool:
    # FAISS L2 distance: lower = more similar.
    # We treat distance > threshold as "not found".
    return best_score > settings.escalation_score_threshold
