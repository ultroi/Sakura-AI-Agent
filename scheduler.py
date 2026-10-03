from __future__ import annotations

from datetime import datetime, time, timezone
import html
import logging
import re
from zoneinfo import ZoneInfo
from telegram.constants import ParseMode

from handlers.helpers import rich_markdown_to_html


logger = logging.getLogger("sakura.scheduler")


def _format_daily_digest(summary: str) -> str:
    """Render the LLM digest into a deterministic Telegram HTML layout."""
    text = (summary or "").replace("\r\n", "\n").strip()

    # Normalize common Markdown heading variants emitted by the model.
    text = re.sub(r"(?im)^\s*#{1,6}\s*today\s*:?\s*$", "[[TODAY]]", text)
    text = re.sub(r"(?im)^\s*(?:📅\s*)?today\s*:?\s*$", "[[TODAY]]", text)
    text = re.sub(r"(?im)^\s*#{1,6}\s*weather\s*:?\s*$", "[[WEATHER]]", text)
    text = re.sub(r"(?im)^\s*(?:🌤️?\s*)?weather\s*:?\s*$", "[[WEATHER]]", text)
    text = re.sub(r"(?im)^\s*#{1,6}\s*notes\s*:?\s*$", "[[NOTES]]", text)
    text = re.sub(r"(?im)^\s*(?:📝\s*)?notes\s*:?\s*$", "[[NOTES]]", text)

    # If the model omitted section headings, keep the text intact rather than
    # inventing content; put it under a single "Update" section.
    sections = {"TODAY": [], "WEATHER": [], "NOTES": []}
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "[[TODAY]]":
            current = "TODAY"
            continue
        if line == "[[WEATHER]]":
            current = "WEATHER"
            continue
        if line == "[[NOTES]]":
            current = "NOTES"
            continue
        if not line:
            if current:
                sections[current].append("")
            continue
        if current:
            sections[current].append(line)

    # Fallback for malformed model output.
    if not any(any(x.strip() for x in values) for values in sections.values()):
        cleaned = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
        sections["TODAY"] = [cleaned.strip()] if cleaned.strip() else ["No updates."]

    def render_lines(lines: list[str]) -> str:
        output = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove duplicated Markdown heading markers.
            line = re.sub(r"^\s*#{1,6}\s*", "", line)

            # Normalize bullet prefixes.
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)

            # Preserve asterisks/backticks as readable text after escaping.
            safe = html.escape(line, quote=False)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)

            # Turn bare URLs into Telegram HTML links.
            safe = re.sub(
                r"(?<![\"=])(https?://[^\s<>]+)",
                lambda m: f'<a href="{html.escape(m.group(1), quote=True)}">link</a>',
                safe,
            )
            output.append(f"• {safe}")
        return "\n".join(output) if output else "• Nothing to report."

    return (
        "<b>🌅 Good Morning, Senpai!</b>\n\n"
        "<b>📅 Today</b>\n"
        f"{render_lines(sections['TODAY'])}\n\n"
        "<b>🌤️ Weather</b>\n"
        f"{render_lines(sections['WEATHER'])}\n\n"
        "<b>📝 Notes</b>\n"
        f"{render_lines(sections['NOTES'])}"
    )


class Scheduler:
    def __init__(self, application, agent):
        self.application = application
        self.agent = agent
        self.tz = ZoneInfo(agent.settings.timezone)
        

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
            await context.bot.send_message(
                chat_id=data["chat_id"],
                text=(
                    "🌸 <b>Yoo-hoo! Ping!</b> ✨\n\n"
                    f"Here is your reminder:\n{data['text']}"
                ),
                parse_mode=ParseMode.HTML,
            )
            await self.agent.reminders.mark_sent(data["reminder_id"])
        except Exception:
            logger.exception("Reminder callback failed | reminder_id=%s", data.get("reminder_id"))

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
        try:
            hour, minute = (int(part) for part in self.agent.settings.digest_time.split(":", 1))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except (TypeError, ValueError):
            logger.warning(
                "Invalid DIGEST_TIME=%r; falling back to 08:00.",
                self.agent.settings.digest_time,
            )
            hour, minute = 8, 0

        self.application.job_queue.run_daily(
            self._daily_digest_callback,
            time=time(hour=hour, minute=minute, tzinfo=self.tz),
            data={"chat_id": owner_chat_id},
            name="daily-digest",
        )

    async def _daily_digest_callback(self, context):
        chat_id = context.job.data["chat_id"]
        try:
            summary = await self.agent.respond(
                telegram_id=chat_id,
                chat_id=chat_id,
                user_text=(
                    "Prepare a concise daily digest for me. Use the available tools to get "
                    "only real, current information when applicable. Return ONLY these sections "
                    "in clean Telegram Markdown, with no introduction, no goodbye, and no anime "
                    "role-play:\n\n"
                    "## Today\n"
                    "- Upcoming calendar events for today or the next 24 hours. If there are none, say 'No calendar events.'\n\n"
                    "## Weather\n"
                    "- A brief current weather snapshot for my saved/local city, only if the city is known. "
                    "If it is not known or weather data is unavailable, say 'Weather unavailable.'\n\n"
                    "## Notes\n"
                    "- 1-3 genuinely useful recent notes/memories, if any. If none, say 'No recent notes.'\n\n"
                    "Rules: do not invent events, weather, locations, notes, dates, or times. "
                    "Prefer short factual bullets. Do not mention tools, APIs, prompts, or internal processing."
                ),
            )

            rendered = _format_daily_digest(summary)
            await context.bot.send_message(
                chat_id=chat_id,
                text=rendered,
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.exception("Morning digest failed | chat_id=%s", chat_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text="🌸 I couldn't prepare the morning digest right now.",
            )

    
    def cancel_reminder(self, reminder_id: str):

        jobs = self.application.job_queue.get_jobs_by_name(f"reminder:{reminder_id}")
        for job in jobs:
            job.schedule_removal() 
