from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from handlers.helpers import is_owner

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update, context):
        await update.message.reply_text("🌸 This is a private assistant.")
        return
    agent = context.application.bot_data["agent"]
    await context.application.bot_data["user_service"].ensure_user(update)
    scheduler = context.application.bot_data.get("scheduler")
    
    if scheduler:
        scheduler.schedule_daily_digest(update.effective_chat.id)
        
    await update.message.reply_text(
        "Konnichiwa! 🌸 I'm <b>Sakura</b>, your personal AI companion!\n\n"
        "Forget those boring, robotic bots—think of me as your trusty sidekick. "
        "Whether you're debugging your ML code, building MERN apps, or just need someone to manage your day, I've got your back! ✨\n\n"
        "Here is a quick peek at my skills:\n"
        "• Fetching live web info, time, and weather instantly. 🌦️\n"
        "• Managing your Google Calendar, Gmail, and GitHub! 💻\n"
        "• Setting quick reminders so you never miss a thing. ⏰\n"
        "• Remembering your preferences and interests so our chats actually feel natural. 🧠\n\n"
        "So, what's our mission for today? Let me know what you need! 🚀",
        parse_mode=ParseMode.HTML
    )


async def connect_google_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update, context):
            await update.message.reply_text("🌸 This is a private assistant.")
            return
    agent = context.application.bot_data["agent"]
    await update.message.reply_text("Starting Google authorization on the machine running Sakura. Complete the browser flow, then come back here.")
    result = await agent.registry.execute("connect_google", {})
    await update.message.reply_text(result)
