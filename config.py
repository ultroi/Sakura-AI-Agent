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
    owner_telegram_id: int
    tavily_api_key: str | None
    github_token: str | None
    timezone: str
    digest_time: str
    owner_telegram_id: int | None
    google_credentials_file: str
    google_token_file: str
    google_maps_api_key: str = ""



def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    owner_id_raw = _required("OWNER_TELEGRAM_ID")
    try:
        owner_id = int(owner_id_raw)
    except ValueError:
        raise ValueError(f"OWNER_TELEGRAM_ID must be a valid integer, got '{owner_id_raw}'")
    return Settings(
        bot_token=_required("BOT_TOKEN"),
        groq_api_key=_required("GROQ_API_KEY"),
        mongodb_uri=_required("MONGODB_URI"),
        google_maps_api_key=_required("GOOGLE_MAPS_API_KEY"),
        mongodb_db=os.getenv("MONGODB_DB", "sakura").strip() or "sakura",
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        github_token=os.getenv("GITHUB_TOKEN", "").strip() or None,
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata",
        digest_time=os.getenv("DIGEST_TIME", "08:00").strip() or "08:00",
        owner_telegram_id=owner_id,
        google_credentials_file=os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json").strip(),
        google_token_file=os.getenv("GOOGLE_TOKEN_FILE", "token.json").strip(),
    )
