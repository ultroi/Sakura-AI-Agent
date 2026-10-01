from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Optional JSON-file bootstrap for headless Google OAuth.
creds_data = os.environ.get("GOOGLE_CREDENTIALS_FILE", "")
if creds_data.lstrip().startswith("{"):
    with open("credentials.json", "w", encoding="utf-8") as f:
        f.write(creds_data)

token_data = os.environ.get("GOOGLE_TOKEN_FILE", "")
if token_data.lstrip().startswith("{"):
    with open("token.json", "w", encoding="utf-8") as f:
        f.write(token_data)

from agent import SakuraAgent
from config import load_settings
from database.connection import MongoDatabase
from database.repositories import UserRepository
from handlers.help import help_command
from handlers.messages import text_handler, voice_handler
from handlers.start import connect_google_command, start_command
from scheduler import Scheduler
from services.user_service import UserService
from utils.logger import setup_logger


async def post_init(application: Application):
    settings = application.bot_data["settings"]
    db = application.bot_data["db"]
    agent = application.bot_data["agent"]

    await db.connect()

    user_repo = UserRepository(db.db)
    application.bot_data["user_service"] = UserService(user_repo)

    agent.bot = application.bot

    scheduler = Scheduler(application, agent)
    application.bot_data["scheduler"] = scheduler

    await scheduler.restore_reminders()

    owner_id = settings.owner_telegram_id or await user_repo.get_owner_id()
    if owner_id:
        scheduler.schedule_daily_digest(owner_id)


async def post_shutdown(application: Application):
    agent = application.bot_data.get("agent")
    if agent:
        await agent.aclose()

    db = application.bot_data.get("db")
    if db:
        await db.close()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger = logging.getLogger("sakura")
    logger.exception("Unhandled Telegram update error", exc_info=context.error)


def build_application() -> Application:
    settings = load_settings()
    db = MongoDatabase(settings)
    agent = SakuraAgent(settings, db.db)

    application = (
        ApplicationBuilder()
        .token(settings.bot_token)
        .connect_timeout(8)
        .read_timeout(45)
        .write_timeout(45)
        .pool_timeout(10)
        .get_updates_connect_timeout(8)
        .get_updates_read_timeout(25)
        .get_updates_write_timeout(25)
        .get_updates_pool_timeout(10)
        .concurrent_updates(4)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    agent.bot = application.bot

    application.bot_data["settings"] = settings
    application.bot_data["db"] = db
    application.bot_data["agent"] = agent

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("connect_google", connect_google_command))
    application.add_handler(
        MessageHandler(
            filters.VOICE & ~filters.COMMAND,
            voice_handler,
        )
    )
    application.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL)
            & ~filters.COMMAND,
            text_handler,
        )
    )
    application.add_error_handler(error_handler)

    return application


def main():
    setup_logger()
    application = build_application()
    application.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
