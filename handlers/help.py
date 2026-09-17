from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌸 <b>Sakura commands</b>\n\n"
        "/start — initialize Sakura\n"
        "/help — show this help\n"
        "/connect_google — authorize Gmail + Calendar\n\n"
        "Most actions are natural language. Just message me normally.",
        parse_mode=ParseMode.HTML,
    )
