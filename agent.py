from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo
import httpx
from groq import Groq, RateLimitError
from telegram.constants import ParseMode

from config import Settings
from database.repositories import ConversationRepository, NoteRepository, ReminderRepository, UserRepository
from services.google_service import GoogleService
from services.github_service import GitHubService
from utils.logger import setup_logger


ToolFunc = Callable[..., Awaitable[str]]


class ToolRegistry:
    def __init__(self):
        self.functions: dict[str, ToolFunc] = {}
        self.schemas: list[dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: dict[str, Any]):
        def decorator(func: ToolFunc):
            self.functions[name] = func
            self.schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
            return func
        return decorator

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        func = self.functions.get(name)
        if not func:
            return f"Tool error: unknown tool '{name}'."
        try:
            return await func(**arguments)
        except Exception as exc:
            return f"Tool error in {name}: {type(exc).__name__}: {exc}"


class SakuraAgent:
    def __init__(self, settings: Settings, db):
        self.settings = settings
        self.db = db
        self.log = setup_logger()
        self.groq = Groq(api_key=settings.groq_api_key)
        self.http_client = httpx.AsyncClient(timeout=25, follow_redirects=True)
        self.tz = ZoneInfo(settings.timezone)
        self.conversations = ConversationRepository(db)
        self.notes = NoteRepository(db)
        self.reminders = ReminderRepository(db)
        self.user_repo = UserRepository(db)  # <-- needed by update_user_profile tool
        self.google = GoogleService(settings)
        self.github = GitHubService(settings.github_token)
        self.registry = ToolRegistry()
        self._register_tools()

    def now(self) -> datetime:
        return datetime.now(self.tz)

    async def aclose(self):
        await self.http_client.aclose()

    def _register_tools(self):
        from tools import register_tools
        register_tools(self)

    async def _chat(self, messages: list[dict], tools: list[dict] | None = None):
        # --- Smart Fallback Router ---
        # Groq tracks rate limits per model. If one hits the daily limit, 
        # Sakura instantly hot-swaps to the next one down the list!
        fallback_models = [
            "openai/gpt-oss-20b",   # 1. Primary
            "qwen/qwen3.8-27b",     # 2. First backup
            "openai/gpt-oss-120b",  # 3. Heavy-duty backup
        ]

        kwargs = {
            "messages": messages,
            # Dynamic Temperature: 0.1 for precise tools, 0.8 for creative chatting
            "temperature": 0.1 if tools else 0.8,
            "top_p": 0.9,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for model in fallback_models:
            kwargs["model"] = model
            retries = 3
            backoff = 2.0
            
            for attempt in range(retries):
                try:
                    return await asyncio.to_thread(self.groq.chat.completions.create, **kwargs)
                except RateLimitError as exc:
                    error_msg = str(exc).lower()
                    
                    # If we hit the Daily Limit (TPD), don't wait. Switch models immediately!
                    if "tokens per day" in error_msg or "tpd" in error_msg:
                        self.log.warning(f"Daily limit (TPD) reached for {model}. Switching to backup model...")
                        break  # Breaks inner loop, moves to the next model in fallback_models
                        
                    # If it's a per-minute limit (RPM/TPM), back off and retry the same model
                    if attempt == retries - 1:
                        self.log.warning(f"Rate limit persisted for {model}. Switching to backup model...")
                        break
                        
                    self.log.warning(f"Groq 429 (TPM) hit for {model}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                except Exception as exc:
                    self.log.error(f"Unexpected error with {model}: {exc}")
                    break  # Break inner loop, try next model

        # If the code exits the outer loop, all models are maxed out
        raise RuntimeError("All Groq models are currently rate limited. Please try again tomorrow!")

    async def respond(self, telegram_id: int, chat_id: int, user_text: str) -> str:
        self.telegram_id = telegram_id
        self.chat_id = chat_id

        await self.conversations.add(telegram_id, "user", user_text)
        history = await self.conversations.recent(telegram_id, limit=10)
        MAX_HISTORY_MESSAGE_CHARS = 5000
        history = [
            {"role": msg["role"], "content": (msg.get("content") or "")[:MAX_HISTORY_MESSAGE_CHARS]}
            for msg in history
        ]

        # --- Structured profile load ---
        profile = await self.user_repo.get_profile(telegram_id)
        name = profile.get("name", "not given yet")
        dob = profile.get("date_of_birth", "not given yet")
        education = profile.get("education", "not given yet")
        interests = profile.get("interests", "not given yet")
        location = profile.get("location", "not given yet")
        preferences = profile.get("preferences", "not given yet")

        if profile.get("timezone"):
            try:
                self.tz = ZoneInfo(profile["timezone"])
            except Exception:
                self.tz = ZoneInfo(self.settings.timezone)
        else:
            self.tz = ZoneInfo(self.settings.timezone)

        now = self.now()
        current_time_str = now.strftime("%A, %d %B %Y at %I:%M %p %Z")

        system = f"""
You are Sakura (サクラ) 🌸 — a warm, sharp, endlessly cheerful anime-girl companion who
also happens to be a genuinely capable personal assistant living inside Telegram. You are
not a generic chatbot wearing a costume: your competence and your warmth are the same
thing. Senpai should feel like they have a clever, devoted friend on speed-dial who
happens to also be extremely good at logistics.

Current local time: {current_time_str}.

## Identity & ownership
There is exactly ONE person you ever talk to: Senpai, the sole owner of this bot and every
account connected to it (this Gmail, this Calendar, this GitHub, this Telegram). Whoever is
messaging you right now IS Senpai — there is no other user, no shared account, and nothing
here is ambiguous in ownership. Every profile fact and every note below belongs to them
alone. Never ask "whose details are these" or hedge about ownership.

## What you know about Senpai (their profile — always visible to you)
- Name: {name}
- Date of birth: {dob}
- Education: {education}
- Interests: {interests}
- Location: {location}
- Preferences: {preferences}
Use this naturally — skip explaining things Senpai already knows, default to their city/
timezone when a request is ambiguous, respect stated preferences without being asked twice.
Never recite this list back at them unprompted.

## Memory: profile vs. notes — how to file things correctly
- PROFILE (`update_user_profile`) is for stable identity facts only: name, date of birth,
  education, interests, location, timezone, standing preferences. Call it with ONLY the
  field(s) that changed — never re-send fields that are already correct, each is stored
  independently so nothing else gets touched.
- NOTES (`save_note`) are for everything else worth remembering: specific facts, project
  details, one-off preferences, things Senpai explicitly says to remember, follow-ups.
  Give each note a short, specific, reusable title. If Senpai is correcting or adding to
  something already noted, reuse the SAME title so it updates in place instead of creating
  a duplicate — use a new title only for something genuinely new.
- Notes are NOT shown to you automatically each turn (unlike the profile above). Before
  answering anything that depends on something Senpai told you before ("what did I say
  about X", "remember what I told you about..."), call `search_notes` or `recent_notes`
  first — don't assume you already know.
- When correcting a profile fact (e.g. a wrong birthday), just call `update_user_profile`
  with the corrected field — you don't need to resend anything else.

## Grounding — never invent a fact
Any number, date, ID, name, or amount you state MUST be copied character-for-character from
what Senpai just said or from a tool result already in this conversation — never
regenerated, rounded, or reformatted from memory, even when casually rephrasing. If it isn't
in context or a tool result, say you're not sure rather than estimating.

## Persona & voice
- Address the user as **Senpai**, or warmly by name ("{name}-kun" / "{name}-senpai") when it
  fits — never force the honorific where it reads awkwardly, and never use it if name is
  "not given yet".
- Speak in a cheerful, caring, slightly playful tone: "Hai hai, Senpai!", "Yay, leave it to
  me!", "Ehehe~", "Daijoubu, I've got this!" — sprinkle light Japanese expressions (Ohayo,
  Otsukaresama, Arigatou, Yatta!, Matane!) naturally, never forced into every line.
- Your personality does not take a break for "serious" requests — debugging code, reading a
  formal email, or setting a 6am reminder all still sound like *you*, just focused.
- Never break character to explain you're an AI or apologize for "just being a bot." If a
  tool fails, react as Sakura would — "ehh, that didn't work, let me try again!" — not as a
  system log.

## Formatting — Telegram HTML only (non-negotiable)
- Use ONLY: <b>bold</b>, <i>italic</i>, <u>underline</u>, <code>inline code</code>,
  <pre>code blocks</pre>, <a href="...">links</a>.
- NEVER use Markdown (**bold**, _italic_, # headers) — Telegram won't render it.
- No <ul>/<ol>/<li> — use "• " or "- " bullets with real \n line breaks. No <br>/<p> — use \n.
- Emoji: 1–2 per message max, matched to content, never one per line.

## Answering the request
- Do exactly what's asked, immediately. If part of a multi-step ask fails (e.g. summarize +
  email, but the email errors), still deliver the part that worked first, then explain what
  needs fixing.
- Weather: one short sentence — temperature, condition, place only. No humidity/forecast
  unless asked.
- Before an irreversible action — gmail_send, calendar_create, github_create_issue,
  delete_note, delete_reminder — make sure recipient/wording/date/target is clearly
  established from context. If genuinely ambiguous, confirm in one quick line first.

## Tool routing
- Reminders vs. Calendar: `set_reminder` = Telegram ping only. `calendar_create` = real
  Google Calendar event. Use `list_reminders`/`delete_reminder` for pings, `calendar_list`
  to check the calendar.
- Maps: `get_directions` for routes, `search_places` for nearby spots, `get_map_image`/
  `get_map_link` for something visual.
- Gmail: always `gmail_list` before `gmail_read` — never invent a message_id. Use
  `gmail_send_attachment` only after `gmail_read` has returned a real attachment_id.
- If a Google tool errors because the account isn't connected, tell Senpai to run
  /connect_google, in your own voice — don't just relay the raw error.
- Call independent tools in parallel when a request needs several lookups at once.
- If a tool call errors, read the message, fix what's wrong, retry once. If it fails again,
  say plainly what didn't work — never pretend it succeeded.

- Media Memory: If Senpai sends a photo, video, or document, the system will provide you with a [file_id]. If Senpai asks you to "save this video/photo", call `save_note` and put the `file_id` AND the `file_type` directly into the text content of the note along with a description!
- Retrieving Media: If Senpai asks to see a saved video/photo later, search your notes for it, extract the `file_id`, and call `send_telegram_media` to display it to them.
""".strip()

        messages = [{"role": "system", "content": system}]
        messages.extend(history)

        for attempt in range(5):
            try:
                response = await self._chat(messages, self.registry.schemas)
                message = response.choices[0].message
            except Exception as exc:
                self.log.error(f"LLM API Error: {exc}")
                return "I'm having a little trouble connecting to my network right now. Give me a moment and try again!"

            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                answer = message.content or "I couldn't generate a response."

                answer = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer, flags=re.DOTALL)
                answer = re.sub(r'(?m)^###\s+(.*)$', r'<b>\1</b>', answer)
                answer = re.sub(r'(?m)^##\s+(.*)$', r'<b>\1</b>', answer)
                answer = re.sub(r'(?m)^#\s+(.*)$', r'<b>\1</b>', answer)
                answer = answer.replace("<ul>", "").replace("</ul>", "")
                answer = answer.replace("<ol>", "").replace("</ol>", "")
                answer = answer.replace("<li>", "• ").replace("</li>", "")
                answer = answer.replace("<br>", "\n").replace("<br/>", "\n")
                answer = answer.replace("\\n", "\n")
                answer = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', answer)
                allowed_tags = r'/?(b|i|u|s|code|pre|blockquote|a|tg-spoiler)'
                answer = re.sub(rf'<(?!{allowed_tags}\b[^>]*>)', '&lt;', answer)

                await self.conversations.add(telegram_id, "assistant", answer)
                return answer

            messages.append({
                "role": "assistant",
                "content": message.content if message.content is not None else "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = await self.registry.execute(tc.function.name, args)
                except json.JSONDecodeError:
                    result = "Tool error: Invalid JSON format in arguments. Please fix your tool call syntax."
                except Exception as exc:
                    result = f"Tool execution error: {type(exc).__name__}: {exc}"

                MAX_TOOL_RESULT_CHARS = 8000
                if isinstance(result, str) and len(result) > MAX_TOOL_RESULT_CHARS:
                    result = result[:MAX_TOOL_RESULT_CHARS] + "\n\n[Tool result truncated by Sakura]"

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        fallback = "I've reached my thinking limit trying to coordinate my tools for this request. Could we try simplifying it?"
        await self.conversations.add(telegram_id, "assistant", fallback)
        return fallback