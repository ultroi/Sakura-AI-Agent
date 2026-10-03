from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from handlers.helpers import is_owner


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update, context):
        await update.message.reply_text("🌸 This is a private assistant.")
        return

    agent = context.application.bot_data["agent"]
    user_id = update.effective_user.id

    await context.application.bot_data["user_service"].ensure_user(update)
    scheduler = context.application.bot_data.get("scheduler")

    # Clear only chat history. Notes, reminders, watches, and profile are kept.
    try:
        await agent.conversations.clear(user_id)
    except AttributeError:
        pass

    if scheduler:
        scheduler.schedule_daily_digest(update.effective_chat.id)

    # Personalize the greeting if we already know the user's name.
    profile = {}
    try:
        profile = await agent.user_repo.get_profile(user_id)
    except Exception:
        pass
    name = (profile.get("name") or "").strip()
    greeting = f"Hey {name}." if name else "Hey Senpai."

    await update.message.reply_text(
        f"{greeting} 🌸\n\n"
        "I'm <b>Sakura</b> — your assistant inside Telegram.\n\n"
        "<b>What I can do</b>\n"
        "• Answer questions, summarize, translate, do math, search the web\n"
        "• Read your Gmail, send mail, manage your Google Calendar\n"
        "• Save notes and remember facts about you across chats\n"
        "• Set reminders and watch pages, products, or anime for changes\n"
        "• Check GitHub, read Google Docs, look up maps and weather\n\n"
        "Just talk to me normally — no commands needed. "
        "Use /help for the short command list, or /connect_google to link Google.",
        parse_mode=ParseMode.HTML,
    )


async def connect_google_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update, context):
        await update.message.reply_text("🌸 This is a private assistant.")
        return
    agent = context.application.bot_data["agent"]
    await update.message.reply_text(
        "Starting Google authorization on the machine running Sakura. "
        "Complete the browser flow, then come back here."
    )
    result = await agent.connect_google()
    await update.message.reply_text(result)