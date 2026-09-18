from __future__ import annotations

import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# --- CLOUD DEPLOYMENT HACK ---
# Stackhost is headless, so we must generate the physical JSON files 
# directly from the Environment Variables before the bot boots!
creds_data = os.environ.get("GOOGLE_CREDENTIALS_FILE", "")
if "{" in creds_data:
    with open("credentials.json", "w", encoding="utf-8") as f:
        f.write(creds_data)

token_data = os.environ.get("GOOGLE_TOKEN_FILE", "")
if "{" in token_data:
    with open("token.json", "w", encoding="utf-8") as f:
        f.write(token_data)
# -----------------------------

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
    agent.schedule_reminder = scheduler.schedule_reminder
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


def build_application() -> Application:
    settings = load_settings()
    db = MongoDatabase(settings)
    agent = SakuraAgent(settings, db.db)

    application = (
        Application.builder()
        .token(settings.bot_token)
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
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL | filters.VOICE) & ~filters.COMMAND, 
        text_handler
    ))

    return application


def main():
    setup_logger()
    application = build_application()
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()