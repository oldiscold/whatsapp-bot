from collections import deque
from typing import List, Dict
from app.config import settings


_sessions: Dict[str, deque] = {}


def get_history(phone: str) -> List[Dict[str, str]]:
    if phone not in _sessions:
        return []
    return list(_sessions[phone])


def add_message(phone: str, role: str, text: str) -> None:
    if phone not in _sessions:
        _sessions[phone] = deque(maxlen=settings.max_history_messages)
    _sessions[phone].append({"role": role, "content": text})
