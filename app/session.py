import json
import os
from collections import deque
from typing import List, Dict

from app.config import settings

_HISTORY_TTL = 7 * 24 * 3600  # 7 дней

# Try Redis, fall back to in-memory if not configured
_redis = None
_sessions: Dict[str, deque] = {}

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis
        _redis = redis.from_url(url, decode_responses=True)
        _redis.ping()
        return _redis
    except Exception:
        return None


def _key(phone: str) -> str:
    return f"history:{phone}"


def get_history(phone: str) -> List[Dict[str, str]]:
    r = _get_redis()
    if r:
        raw = r.get(_key(phone))
        if raw:
            return json.loads(raw)
        return []
    # fallback in-memory
    if phone not in _sessions:
        return []
    return list(_sessions[phone])


def add_message(phone: str, role: str, text: str) -> None:
    r = _get_redis()
    if r:
        history = get_history(phone)
        history.append({"role": role, "content": text})
        if len(history) > settings.max_history_messages:
            history = history[-settings.max_history_messages:]
        r.set(_key(phone), json.dumps(history, ensure_ascii=False), ex=_HISTORY_TTL)
        return
    # fallback in-memory
    if phone not in _sessions:
        _sessions[phone] = deque(maxlen=settings.max_history_messages)
    _sessions[phone].append({"role": role, "content": text})
