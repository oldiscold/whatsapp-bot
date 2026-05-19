import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response

from app import session, whatsapp_client
from app.config import settings
from app.escalation import build_escalation_reply, check_keywords, check_rag_score
from app.rag_chain import get_answer, search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NON_TEXT_REPLY = "Я могу отвечать только на текстовые сообщения. Напишите ваш вопрос текстом 😊"

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.verify_token:
        return Response(content=challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()

    # Validate HMAC-SHA256 signature
    signature_header = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        settings.app_secret.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        entry = body["entry"][0]["changes"][0]["value"]
        message = entry["messages"][0]
        phone = message["from"]
        msg_type = message.get("type", "unknown")
        text = message.get("text", {}).get("body", "") if msg_type == "text" else ""
    except (KeyError, IndexError):
        return Response(status_code=200)

    background_tasks.add_task(_process_message, phone, msg_type, text)
    return Response(status_code=200)


async def _process_message(phone: str, msg_type: str, text: str) -> None:
    start = time.monotonic()
    phone_hash = "sha256:" + hashlib.sha256(phone.encode()).hexdigest()[:16]

    if msg_type != "text":
        await whatsapp_client.send_message(phone, NON_TEXT_REPLY)
        _log(phone_hash, 0, len(NON_TEXT_REPLY), time.monotonic() - start, "non_text", None, 0)
        return

    # Stage 1: keyword escalation (cheap, before RAG)
    if check_keywords(text):
        reply = build_escalation_reply()
        await whatsapp_client.send_message(phone, reply)
        _log(phone_hash, len(text), len(reply), time.monotonic() - start, "escalation_keywords", None, 0)
        return

    # RAG search
    chunks, best_score = search(text)

    # Stage 2: RAG score escalation
    if check_rag_score(best_score):
        reply = build_escalation_reply()
        await whatsapp_client.send_message(phone, reply)
        _log(phone_hash, len(text), len(reply), time.monotonic() - start, "escalation_rag", best_score, 0)
        return

    # Full pipeline
    history = session.get_history(phone)
    session.add_message(phone, "user", text)

    reply = await get_answer(text, history)

    session.add_message(phone, "assistant", reply)
    await whatsapp_client.send_message(phone, reply)
    _log(phone_hash, len(text), len(reply), time.monotonic() - start, "rag+gpt", best_score, 3)


def _log(
    phone_hash: str,
    msg_len: int,
    resp_len: int,
    elapsed: float,
    proc_type: str,
    rag_score: float | None,
    fewshot_count: int,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phone_hash": phone_hash,
        "message_length": msg_len,
        "response_length": resp_len,
        "latency_ms": round(elapsed * 1000),
        "processing_type": proc_type,
        "rag_top_score": rag_score,
        "fewshot_count": fewshot_count,
    }
    logger.info(json.dumps(record, ensure_ascii=False))
