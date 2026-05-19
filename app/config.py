import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    whatsapp_token: str = Field(..., env="WHATSAPP_TOKEN")
    phone_number_id: str = Field(..., env="PHONE_NUMBER_ID")
    verify_token: str = Field(..., env="VERIFY_TOKEN")
    app_secret: str = Field(..., env="APP_SECRET")

    escalation_score_threshold: float = Field(0.7, env="ESCALATION_SCORE_THRESHOLD")
    max_history_messages: int = Field(8, env="MAX_HISTORY_MESSAGES")
    max_prompt_tokens: int = Field(6000, env="MAX_PROMPT_TOKENS")
    escalation_contact: str = Field(
        "+7 (xxx) xxx-xx-xx или @username в Telegram",
        env="ESCALATION_CONTACT",
    )

    bot_name: str = Field("Бек", env="BOT_NAME")
    company_name: str = Field("JV Team", env="COMPANY_NAME")

    faiss_index_path: str = "data/faiss_index"
    fewshot_index_path: str = "data/fewshot_index"
    fewshot_pairs_path: str = "data/fewshot_pairs.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
