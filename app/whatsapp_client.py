import asyncio
import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

_API_URL = "https://graph.facebook.com/v19.0/{phone_number_id}/messages"


async def send_message(to: str, text: str) -> None:
    url = _API_URL.format(phone_number_id=settings.phone_number_id)
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    delays = [1, 2, 4]
    async with httpx.AsyncClient(timeout=10) as client:
        for attempt, delay in enumerate(delays, 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    logger.warning("WhatsApp API attempt %d: status %d body: %s", attempt, resp.status_code, resp.text)
                    if attempt < len(delays):
                        await asyncio.sleep(delay)
                    continue
                if not resp.is_success:
                    logger.error("WhatsApp API error %d: %s", resp.status_code, resp.text)
                    return
                resp.raise_for_status()
                return
            except httpx.HTTPError as exc:
                logger.warning("WhatsApp API attempt %d: %s", attempt, exc)
                if attempt < len(delays):
                    await asyncio.sleep(delay)

    logger.error("Failed to send WhatsApp message to %s after %d attempts", to, len(delays))
