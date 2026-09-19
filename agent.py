from __future__ import annotations

import asyncio
import html
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


# Telegram Rich Messages support Rich Markdown directly.  Using Markdown as the
# model's output contract is much more reliable than asking the model to invent
# HTML tags and then trying to repair malformed nesting afterwards.
_ALLOWED_INLINE_HTML = {
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "code", "mark", "sub", "sup", "tg-spoiler", "a", "details",
    "summary", "tg-math", "tg-math-block", "tg-reference", "tg-emoji",
    "tg-time",
}


def normalize_rich_markdown(text: str) -> str:
    """
    Deterministically clean model output before passing it to Telegram's
    Rich Markdown parser.

    The model is not trusted to author Telegram HTML. Markdown is used as the
    output contract because Telegram parses headings/lists/tables/quotes/code
    natively in Rich Messages.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Protect fenced code blocks before touching HTML or bullet separators.
    fenced: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        fenced.append(match.group(0))
        return f"\n@@__SAKURA_CODE_{len(fenced)-1}__@@\n"

    text = re.sub(r"```[\s\S]*?```", stash_code, text)

    # Remove unsupported HTML tags outside code. The model should use Markdown
    # for normal formatting, not arbitrary HTML tags.
    def clean_tag(match: re.Match[str]) -> str:
        raw = match.group(0)
        m = re.match(r"</?\s*([A-Za-z0-9-]+)", raw)
        if not m:
            return raw
        name = m.group(1).lower()
        return raw if name in _ALLOWED_INLINE_HTML else ""

    text = re.sub(
        r"</?\s*[A-Za-z0-9-]+(?:\s+[^<>]*?)?/?>",
        clean_tag,
        text,
    )

    # Repair a common model failure visible in long assistant replies: several
    # logical bullet items get concatenated into one paragraph with ` • `.
    paragraphs = re.split(r"(\n\s*\n)", text)
    repaired: list[str] = []

    for part in paragraphs:
        if part == "\n\n" or not part.strip():
            repaired.append(part)
            continue

        bullet_count = len(re.findall(r"\s+•\s+", part))
        if bullet_count >= 2 and "@@__SAKURA_CODE_" not in part:
            part = re.sub(r"\s+•\s+", "\n- ", part)

        repaired.append(part)

    text = "".join(repaired)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()

    # Restore fenced code exactly.
    for i, block in enumerate(fenced):
        text = text.replace(
            f"@@__SAKURA_CODE_{i}__@@",
            block.strip("\n"),
        )

    return text


def rich_markdown_to_plain(text: str) -> str:
    """Safe plain-text fallback when Rich Messages are unavailable."""
    if not text:
        return ""

    text = re.sub(r"```(?:[A-Za-z0-9_+.-]+)?\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{4,}", "\n\n", text).strip()


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
        # max_retries=0 ensures Groq does NOT sleep internally for 30+ seconds on 429
        self.groq = Groq(api_key=settings.groq_api_key, max_retries=0)
        self.http_client = httpx.AsyncClient(timeout=25, follow_redirects=True)
        self.tz = ZoneInfo(settings.timezone)
        self.conversations = ConversationRepository(db)
        self.notes = NoteRepository(db)
        self.reminders = ReminderRepository(db)
        self.user_repo = UserRepository(db)
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

    async def _chat_gemini_fallback(self, messages: list[dict]):
        """Direct fallback to Gemini Flash when all Groq free models hit TPM/TPD limits."""
        api_key = getattr(self.settings, "gemini_api_key", None)
        if not api_key:
            raise RuntimeError("Gemini API key not configured for fallback.")

        # Flatten messages into a clean conversational script for Gemini
        conversation_text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                conversation_text += f"\n[{role.upper()}]: {content}\n"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": conversation_text}]}],
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048}
        }

        resp = await self.http_client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini fallback error {resp.status_code}: {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates or "content" not in candidates[0]:
            raise RuntimeError("Gemini fallback returned an empty or blocked candidate.")

        text = candidates[0]["content"]["parts"][0]["text"]

        # Mimic Groq's SDK structure so respond() processes the output without modification
        class DummyChoice:
            def __init__(self, content):
                self.message = DummyMessage(content)

        class DummyMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None

        class DummyResponse:
            def __init__(self, content):
                self.choices = [DummyChoice(content)]

        return DummyResponse(text)

    async def _chat(self, messages: list[dict], tools: list[dict] | None = None):
        fallback_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

        kwargs = {
            "messages": messages,
            "temperature": 0.1 if tools else 0.8,
            "top_p": 0.9,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for model in fallback_models:
            kwargs["model"] = model
            retries = 2
            backoff = 1.5
            
            for attempt in range(retries):
                try:
                    return await asyncio.to_thread(self.groq.chat.completions.create, **kwargs)
                except RateLimitError as exc:
                    error_msg = str(exc).lower()
                    
                    # Daily limit reached -> swap model immediately
                    if "tokens per day" in error_msg or "tpd" in error_msg:
                        self.log.warning(f"Daily limit (TPD) reached for {model}. Switching model...")
                        break
                    
                    self.log.warning(f"Groq 429 (TPM) on {model}. Attempt {attempt + 1}/{retries}...")
                    if attempt == retries - 1:
                        self.log.warning(f"Exhausted retries for {model}, swapping to next fallback.")
                        break
                        
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

                except Exception as exc:
                    error_msg = str(exc).lower()
                    if "413" in error_msg or "payload too large" in error_msg:
                        self.log.warning(f"Payload too large for {model}. Switching model...")
                        break
                    self.log.error(f"Unexpected error with {model}: {exc}")
                    break

        # If all Groq options fail due to TPM or TPD exhaustion, fall back directly to Gemini Flash
        self.log.warning("All Groq models failed or rate-limited. Activating Gemini Flash fallback...")
        try:
            return await self._chat_gemini_fallback(messages)
        except Exception as exc:
            self.log.error(f"Gemini fallback also failed: {exc}")
            raise RuntimeError("All LLM providers are currently rate limited or unreachable.")

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
- MEDIA MEMORY (SAVE): If Senpai sends a photo, video, document, or voice note, the system will
  provide you with a [file_id]. If Senpai asks you to save it, call `save_note` and put the 
  `file_id` AND the `file_type` directly into the text content of the note along with a description!
- MEDIA MEMORY (RETRIEVE): If Senpai asks to see a saved video/photo, search your notes 
  for it, extract the `file_id`, and call `send_telegram_media` to display it to them.
- MANIPULATE & DELETE: You have FULL access to manage all saved data. If Senpai asks you 
  to delete, update, or change a saved note or media file, you MUST first call `search_notes` 
  or `recent_notes` to find its exact `note_id`. Then, call `delete_note` or `update_note` to execute the request.

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

## Dynamic Output & Tool Formatting Rules (CRITICAL)

Differentiate clearly between what is displayed in Telegram vs what is passed into tool arguments.

1. FOR TELEGRAM CHAT DISPLAY (Direct messages to Senpai):
You MUST output <b>Telegram Rich Markdown</b>. Do NOT generate ordinary Telegram HTML as the primary format.
Telegram's Rich Message parser natively supports headings, lists, tables, blockquotes, details blocks, fenced code, inline formatting, and formulas.

Formatting rules:
• Casual/simple replies: keep formatting light with **bold**, _italic_, `inline code`, and links.
• Explanations/tutorials: use # / ## headings, short paragraphs, and real Markdown lists.
• Structured comparisons/data: use a Markdown table when it is genuinely clearer.
• Long document/PDF analysis: use ## / ### sections and <details><summary>...</summary>...</details> for optional deeper material.
• Mathematics: use <tg-math-block>LaTeX</tg-math-block> for standalone equations and <tg-math>...</tg-math> inline when useful.
• Important text: **bold**, ==marked text==, and > blockquotes may be used sparingly.
• Code: ALWAYS use fenced code blocks with the language when known, e.g. ```python ... ```.

HARD RULES:
• Never output explanations about formatting like "I'll use proper formatting". Just format the answer.
• Never invent HTML tags such as <div>, <section>, <table-row>, <span>, or custom tags.
• Do not manually write <table> HTML; use a Markdown table.
• Do not manually write <ul>, <ol>, or <li>; use Markdown lists.
• The only HTML allowed in normal Rich Markdown is the specific Telegram extensions that Markdown cannot express cleanly, especially <details>, <summary>, <tg-math>, and <tg-math-block>.
• Keep every fenced code block intact and never apply formatting inside it.
• Use blank lines between logical sections.
• Use one list item per line. Never concatenate multiple list items into one paragraph.
• Do not put several independent sections on the same line.
• Use a table only when there are at least 2 columns and 2 meaningful rows; otherwise use bullets.
• For document summaries, prefer short sections over one giant paragraph.

2. FOR EXTERNAL TOOLS (Gmail, Notes, GitHub, Calendar, Reminders):
When generating text inside tool arguments (e.g., `body` for gmail_send, `content` for save_note, `title`/`body` for github_create_issue, or `text` for set_reminder):
• DO NOT use Telegram HTML tags.
• Send clean, professional PLAIN TEXT with natural line breaks.
• For Gmail: Write standard, professional email text without Telegram formatting tags.
• For GitHub issues: You MAY use standard Markdown because GitHub natively supports Markdown.

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
- VISION / IMAGES (STRICT RULE): When Senpai sends a photo, you receive a [file_id]. READ THE CAPTION FIRST! 
  If Senpai says "save this" or just gives a title, DO NOT call `analyze_image` — just call `save_note`.
  ONLY call `analyze_image` if Senpai explicitly asks you to explain, read, or analyze what is INSIDE the image.
- DOCUMENTS / FILES (STRICT RULE): When Senpai sends a document, PDF, Word, or Excel file, you receive a [file_id] and [file_name]. 
  ALWAYS READ THE CAPTION FIRST!
  If Senpai says "save this", gives a title, or only wants to store it:
  • DO NOT call `analyze_document`.
  • Call `save_note` and record the `file_id`, `file_name`, and `file_type` ("document").
  ONLY call `analyze_document` when Senpai explicitly instructs you to inspect, read, summarize, solve, or extract information from that document!
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
- TELEGRAM METADATA & FORWARDS: If Senpai forwards a message or asks "who sent this", "get the channel ID", "check who forwarded this", or asks for sender IDs/usernames, call `inspect_telegram_context` to inspect the forwarded origin details.
- RICH TELEGRAM OUTPUT: Return Telegram Rich Markdown, not ordinary Telegram HTML. For document summaries, technical explanations, comparisons, structured data, or formulas, prefer semantic Markdown headings, lists, tables, blockquotes, fenced code, details blocks, and Telegram math tags. Keep casual replies simple.
- INLINE BUTTONS (STRICT RESTRAINT): NEVER use `send_inline_keyboard` for normal chatter, casual replies, or simple text questions. 
  ONLY call `send_inline_keyboard` when:
  1. A critical action needs binary confirmation (e.g. [ Confirm ] / [ Cancel ]).
  2. Providing direct web navigation buttons (e.g. opening Google Maps, GitHub repo links).
  3. Explicitly asked by Senpai to provide interactive choices.
  If none of these apply, reply using ordinary Telegram Rich Markdown text.
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

                # -----------------------------------------------------------------
                # Telegram Rich Markdown normalization
                # -----------------------------------------------------------------
                # The model returns Markdown; Telegram performs the final Rich
                # formatting. This avoids malformed/hallucinated HTML tags.
                answer = normalize_rich_markdown(answer)

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

                MAX_TOOL_RESULT_CHARS = 3000  
                if isinstance(result, str) and len(result) > MAX_TOOL_RESULT_CHARS:
                    result = result[:MAX_TOOL_RESULT_CHARS] + "\n\n[Tool result truncated by Sakura to save memory]"

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        fallback = "I've reached my thinking limit trying to coordinate my tools for this request. Could we try simplifying it?"
        await self.conversations.add(telegram_id, "assistant", fallback)
        return fallback