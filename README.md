# WhatsApp Sales Bot

WhatsApp-бот для ответов на вопросы клиентов в стиле живых менеджеров по продажам.

## Stack
- Python 3.11, FastAPI, LangChain, FAISS, OpenAI GPT-4o
- WhatsApp Cloud API (Meta)

## Быстрый старт

### 1. Настройка окружения
```bash
cp .env.example .env
# Заполните .env своими ключами
```

### 2. Подготовка данных
```bash
# Положить файлы базы знаний в data/product_docs/ (.md или .docx)
# Положить ZIP-архивы чатов в data/dialogs/
# Имена менеджеров определяются автоматически — участник, который
# присутствует во всех ZIP-архивах, считается менеджером.

python scripts/parse_dialogs.py     # → data/fewshot_pairs.json
python scripts/index_documents.py   # → data/faiss_index/
python scripts/index_dialogs.py     # → data/fewshot_index/
```

### 3. Запуск (локально)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Docker
```bash
docker build -t whatsapp-bot .
docker run --env-file .env -p 8000:8000 whatsapp-bot
```

## Настройка webhook в Meta Developer Console
URL: `https://your-domain/webhook`  
Verify token: значение из `VERIFY_TOKEN` в `.env`

## Переменные окружения
| Переменная | Описание |
|---|---|
| `OPENAI_API_KEY` | Ключ OpenAI |
| `WHATSAPP_TOKEN` | Bearer-токен WhatsApp Cloud API |
| `PHONE_NUMBER_ID` | ID номера телефона в Meta |
| `VERIFY_TOKEN` | Токен верификации webhook |
| `APP_SECRET` | App Secret для проверки подписи |
| `ESCALATION_CONTACT` | Контакт для эскалации |
| `ESCALATION_SCORE_THRESHOLD` | Порог RAG score (default: 0.7) |
| `MAX_HISTORY_MESSAGES` | Глубина истории диалога (default: 8) |
| `MAX_PROMPT_TOKENS` | Лимит токенов промпта (default: 6000) |

## Архитектура обработки сообщения
1. Верификация подписи HMAC-SHA256
2. Фильтрация нетекстовых сообщений
3. Проверка эскалации по ключевым словам
4. RAG поиск в FAISS (top-3 чанка)
5. Проверка эскалации по RAG score
6. Подбор Few-Shot примеров из индекса диалогов
7. Сборка промпта с контролем токенов
8. Вызов GPT-4o
9. Отправка ответа с retry
