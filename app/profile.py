import json
import os
from typing import Dict

_PROFILE_TTL = 30 * 24 * 3600  # 30 дней

_profiles: Dict[str, dict] = {}
_redis = None


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
    return f"profile:{phone}"


def get_profile(phone: str) -> dict:
    r = _get_redis()
    if r:
        raw = r.get(_key(phone))
        return json.loads(raw) if raw else {}
    return _profiles.get(phone, {})


def update_profile(phone: str, updates: dict) -> None:
    profile = get_profile(phone)
    profile.update({k: v for k, v in updates.items() if v})
    r = _get_redis()
    if r:
        r.set(_key(phone), json.dumps(profile, ensure_ascii=False), ex=_PROFILE_TTL)
    else:
        _profiles[phone] = profile


def format_profile(profile: dict) -> str:
    if not profile:
        return "Данных пока нет."
    labels = {
        "city": "Город",
        "position": "Должность",
        "niche": "Ниша/сфера",
        "goals": "Задачи/цели",
        "name": "Имя",
    }
    lines = [f"{labels.get(k, k)}: {v}" for k, v in profile.items() if v]
    return "\n".join(lines) if lines else "Данных пока нет."
