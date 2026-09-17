from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from telegram.constants import ParseMode


class Scheduler:
    def __init__(self, application, agent):
        self.application = application
        self.agent = agent
        self.tz = ZoneInfo(agent.settings.timezone)
        
        # Bind these methods to the agent so tools.py can trigger them easily!
        self.agent.schedule_reminder = self.schedule_reminder
        self.agent.cancel_reminder = self.cancel_reminder

    def schedule_reminder(self, reminder_id: str, chat_id: int, text: str, when: datetime):
        # Escape user text to prevent accidental HTML injection crashing the reminder
        text = (text or "").replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.application.job_queue.run_once(
            self._reminder_callback,
            when=when,
            data={"reminder_id": reminder_id, "chat_id": chat_id, "text": text},
            name=f"reminder:{reminder_id}",
        )

    async def _reminder_callback(self, context):
        data = context.job.data
        try:
            # IMPROVEMENT: Anime-style playful reminder text
            await context.bot.send_message(
                chat_id=data["chat_id"], 
                text=f"🌸 <b>Yoo-hoo! Ping!</b> ✨\n\nHere is your reminder:\n{data['text']}", 
                parse_mode=ParseMode.HTML
            )
            await self.agent.reminders.mark_sent(data["reminder_id"])
        except Exception:
            pass

    async def restore_reminders(self):
        now = datetime.now(timezone.utc)
        for reminder in await self.agent.reminders.due_or_pending():
            when = reminder["run_at"]
            text = reminder.get("text") or ""
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when <= now:
                self.application.job_queue.run_once(
                    self._reminder_callback,
                    when=1,
                    data={"reminder_id": str(reminder["_id"]), "chat_id": reminder["chat_id"], "text": text},
                    name=f"reminder:{reminder['_id']}",
                )
            else:
                self.schedule_reminder(str(reminder["_id"]), reminder["chat_id"], text, when.astimezone(self.tz))

    def schedule_daily_digest(self, owner_chat_id: int | None):
        if not owner_chat_id:
            return
        if self.application.job_queue.get_jobs_by_name("daily-digest"):
            return
        hour, minute = (int(x) for x in self.agent.settings.digest_time.split(":", 1))
        self.application.job_queue.run_daily(
            self._daily_digest_callback,
            time=time(hour=hour, minute=minute, tzinfo=self.tz),
            data={"chat_id": owner_chat_id},
            name="daily-digest",
        )

    async def _daily_digest_callback(self, context):
        chat_id = context.job.data["chat_id"]
        try:
            self.agent.telegram_id = chat_id
            self.agent.chat_id = chat_id
            
            # IMPROVEMENT: explicitly instruct the LLM to use the persona for the digest
            summary = await self.agent.respond(
                telegram_id=chat_id,
                chat_id=chat_id,
                user_text=(
                    "Create my morning digest. Briefly combine my upcoming Google Calendar events, "
                    "a useful weather snapshot for my local city if known from context, and any recent notes. "
                    "Format it beautifully using Telegram HTML tags (<b>, <i>, etc.). Do not invent details.\n\n"
                    "IMPORTANT: Stay in character! Write this digest as Sakura, my bright and playful anime-style companion. "
                    "Start with a cheerful 'Ohayou!' or a warm morning greeting! ✨"
                ),
            )
            
            # Use standard \n instead of invalid <br> tags, and enable HTML ParseMode
            await context.bot.send_message(
                chat_id=chat_id, 
                text=f"🌅 <b>Good Morning!</b> 🌸\n\n{summary}",
                parse_mode=ParseMode.HTML
            )
        except Exception as exc:
            await context.bot.send_message(chat_id=chat_id, text=f"Morning digest error: {exc}")

    
    def cancel_reminder(self, reminder_id: str):
        # Find the job by the unique name we gave it during creation
        jobs = self.application.job_queue.get_jobs_by_name(f"reminder:{reminder_id}")
        for job in jobs:
            job.schedule_removal() # This safely terminates the background process!