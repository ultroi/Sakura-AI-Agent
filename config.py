from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    bot_token: str
    groq_api_key: str
    mongodb_uri: str
    mongodb_db: str
    gemini_api_key: str | None
    github_token: str | None
    tavily_api_key: str | None
    timezone: str
    digest_time: str
    owner_telegram_id: int
    google_credentials_file: str
    google_token_file: str


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        bot_token=_required("BOT_TOKEN"),
        groq_api_key=_required("GROQ_API_KEY"),
        mongodb_uri=_required("MONGODB_URI"),
        mongodb_db=os.getenv("MONGODB_DB", "sakura").strip() or "sakura",
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        github_token=os.getenv("GITHUB_TOKEN", "").strip() or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata",
        digest_time=os.getenv("DIGEST_TIME", "08:00").strip() or "08:00",
        owner_telegram_id=int(_required("OWNER_TELEGRAM_ID")),
        google_credentials_file=os.getenv(
            "GOOGLE_CREDENTIALS_FILE", "credentials.json"
        ).strip(),
        google_token_file=os.getenv("GOOGLE_TOKEN_FILE", "token.json").strip(),
    )
