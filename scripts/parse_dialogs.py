"""Parse WhatsApp ZIP exports into fewshot_pairs.json.

Manager auto-detection: the sender who appears in ALL (or most) ZIP archives
is the manager — same person across different client chats.

Supported WhatsApp export formats:
  [DD.MM.YYYY, HH:MM] Name: text   (bracketed, older exports)
  DD.MM.YYYY, HH:MM - Name: text   (dash-separated, newer exports)
"""
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

DIALOGS_DIR = Path("data/dialogs")
OUTPUT_FILE = Path("data/fewshot_pairs.json")

MEDIA_RE = re.compile(
    r"<Медиа отсутствует>|<Media omitted>|image omitted|sticker omitted"
    r"|video omitted|audio omitted|document omitted"
    r"|\(файл добавлен\)|\(file attached\)",
    re.IGNORECASE,
)
EDITED_RE = re.compile(r"\s*<Сообщение изменено>", re.IGNORECASE)

# Two supported formats
MSG_BRACKET = re.compile(r"^\[(\d{2}\.\d{2}\.\d{4}), (\d{2}:\d{2})\] ([^:]+): (.+)$")
MSG_DASH    = re.compile(r"^(\d{2}\.\d{2}\.\d{4}), (\d{2}:\d{2}) - ([^:]+): (.+)$")
# System message (no "Name: text" part)
SYS_BRACKET = re.compile(r"^\[\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2}\] [^:]+$")
SYS_DASH    = re.compile(r"^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2} - [^:]+$")


def _match_msg(line: str):
    """Return (sender, body) or None."""
    for pattern in (MSG_DASH, MSG_BRACKET):
        m = pattern.match(line)
        if m:
            return m.group(3).strip(), m.group(4).strip()
    return None


def _is_system(line: str) -> bool:
    return bool(SYS_DASH.match(line) or SYS_BRACKET.match(line))


def read_chat_txt(zip_path: Path) -> str | None:
    with zipfile.ZipFile(zip_path) as zf:
        txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_files:
            return None
        return zf.read(txt_files[0]).decode("utf-8", errors="replace")


def collect_senders(text: str) -> set[str]:
    senders = set()
    for line in text.splitlines():
        result = _match_msg(line)
        if result:
            senders.add(result[0])
    return senders


def detect_managers(zip_files: list[Path]) -> set[str]:
    """Names present in ALL ZIPs are managers. Falls back to majority (>50%)."""
    counter: Counter = Counter()
    total = len(zip_files)

    for zp in zip_files:
        text = read_chat_txt(zp)
        if text is None:
            continue
        for name in collect_senders(text):
            counter[name] += 1

    threshold = total if total <= 3 else total * 0.6
    managers = {name for name, cnt in counter.items() if cnt >= threshold}

    if not managers:
        print("Не удалось автоматически определить менеджера.")
        print("Все отправители:", sorted(counter))
        raw = input("Введите имена менеджеров через запятую: ")
        managers = {n.strip() for n in raw.split(",") if n.strip()}

    return {m.lower() for m in managers}


def parse_chat(text: str, manager_names: set[str]) -> list[dict]:
    messages = []
    current = None

    for line in text.splitlines():
        if _is_system(line):
            current = None
            continue

        result = _match_msg(line)
        if result:
            sender, body = result
            if current:
                messages.append(current)
            current = None

            # Filter media/file lines
            if MEDIA_RE.search(body):
                continue

            body = EDITED_RE.sub("", body).strip()
            if not body:
                continue

            role = "manager" if sender.lower() in manager_names else "client"
            current = {"role": role, "text": body}

        elif current and line.strip():
            # Continuation of previous message (no timestamp prefix)
            extra = EDITED_RE.sub("", line).strip()
            if extra and not MEDIA_RE.search(extra):
                current["text"] += "\n" + extra

    if current:
        messages.append(current)
    return messages


def extract_pairs(messages: list[dict]) -> list[dict]:
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "client":
            client_parts = []
            while i < len(messages) and messages[i]["role"] == "client":
                client_parts.append(messages[i]["text"])
                i += 1
            manager_parts = []
            while i < len(messages) and messages[i]["role"] == "manager":
                manager_parts.append(messages[i]["text"])
                i += 1
            if manager_parts:
                manager_text = " ".join(manager_parts)
                if len(manager_text) >= 5:
                    pairs.append({
                        "client": " ".join(client_parts),
                        "manager": manager_text,
                    })
        else:
            i += 1
    return pairs


def main():
    zip_files = sorted(DIALOGS_DIR.glob("*.zip"))
    if not zip_files:
        print(f"ZIP-файлы не найдены в {DIALOGS_DIR}")
        return

    print(f"Найдено ZIP-архивов: {len(zip_files)}")
    manager_names = detect_managers(zip_files)
    print(f"Определены менеджеры: {manager_names}")

    all_pairs = []
    for zp in zip_files:
        text = read_chat_txt(zp)
        if text is None:
            print(f"  .txt файл не найден внутри {zp.name}")
            continue
        messages = parse_chat(text, manager_names)
        pairs = extract_pairs(messages)
        all_pairs.extend(pairs)
        print(f"  {zp.name}: {len(pairs)} пар")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\nВсего пар: {len(all_pairs)}")
    print(f"Сохранено в: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
