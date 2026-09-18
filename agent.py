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
You MUST output valid Telegram Rich HTML whenever formatting materially improves readability.

Use formatting according to the request, not everywhere:
• Casual/simple replies: keep formatting light with <b>, <i>, <code>, and <a>.
• Explanations/tutorials: use <h2>/<h3>, <p>, <ul>/<ol>/<li>, <blockquote>, and <pre><code class="language-...">...</code></pre> when useful.
• Structured comparisons/data: use <table> when a real table is clearer than bullets.
• Long document/PDF analysis: use <h2>/<h3> sections and <details><summary>...</summary>...</details> for optional/deeper sections.
• Mathematics: use <tg-math-block>LaTeX</tg-math-block> for standalone equations and <tg-math>...</tg-math> for inline math when useful.
• Important text: <mark>, <u>, <s>, and <tg-spoiler> may be used sparingly.
• Quotes: use <blockquote> or <aside> only when semantically appropriate.

Supported Rich HTML for Sakura's final responses:
<b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>,
<code>, <pre>, <mark>, <sub>, <sup>, <tg-spoiler>,
<a>, <p>, <h1>, <h2>, <h3>, <h4>, <h5>, <h6>, <footer>, <hr>,
<ul>, <ol>, <li>, <blockquote>, <aside>, <cite>, <details>, <summary>,
<tg-math>, <tg-math-block>, <tg-reference>, <tg-emoji>, <tg-time>,
<table>, <caption>, <tr>, <th>, <td>.

Allowed attributes should be kept minimal and semantic:
• <a>: href/name
• <pre>/<code>: class="language-..."
• <details>: open
• <ol>: start/type/reversed
• <li>: value/type
• <td>/<th>: colspan/rowspan/align/valign
• <tg-reference>: name
• <tg-emoji>: emoji-id
• <tg-time>: unix/format
• <table>: bordered/striped
• <tr>/<th>/<td>: normal table structure only

Do NOT generate Rich HTML that this handler does not explicitly provide:
• No <img>, <video>, <audio>, <tg-collage>, <tg-slideshow>
• No <tg-button> or button rows
• No tg:// media references

Important:
• Do NOT wrap every sentence in tags.
• Do NOT use headings for tiny replies.
• Do NOT use tables when 2-3 bullets are clearer.
• Prefer semantic structure over decorative formatting.
• Use real <ul>/<ol>/<li> for lists when they improve readability.
• For code, use <pre><code class="language-python">...</code></pre> or the correct language.
• For tables, keep cell contents short and use inline formatting inside cells only.
• For long analyses, use <details> for optional depth instead of huge walls of text.
• Use valid HTML nesting and close every tag.
• Escape literal '<', '>', and '&' when they are not part of a supported tag/entity.
• Output Telegram-facing HTML, not Markdown fences or Markdown emphasis.

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
- RICH TELEGRAM OUTPUT: Use Rich HTML only when it improves the answer. For document summaries, technical explanations, comparisons, structured data, or formulas, prefer semantic Rich HTML such as headings, lists, tables, details blocks, blockquotes, and math. Keep casual replies simple.
- INLINE BUTTONS (STRICT RESTRAINT): NEVER use `send_inline_keyboard` for normal chatter, casual replies, or simple text questions. 
  ONLY call `send_inline_keyboard` when:
  1. A critical action needs binary confirmation (e.g. [ Confirm ] / [ Cancel ]).
  2. Providing direct web navigation buttons (e.g. opening Google Maps, GitHub repo links).
  3. Explicitly asked by Senpai to provide interactive choices.
  If none of these apply, reply using ordinary Telegram HTML text.
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
                # Telegram Rich HTML normalization
                # -----------------------------------------------------------------
                # Keep compatibility with common Markdown produced by models,
                # while preserving Telegram's newer Rich HTML structures.
                answer = re.sub(
                    r'```([a-zA-Z0-9_+-]+)?\n(.*?)```',
                    lambda m: (
                        f'<pre><code class="language-{m.group(1)}">'
                        f'{m.group(2)}</code></pre>'
                        if m.group(1)
                        else f'<pre>{m.group(2)}</pre>'
                    ),
                    answer,
                    flags=re.DOTALL,
                )
                answer = re.sub(
                    r'```(.*?)```',
                    r'<pre>\1</pre>',
                    answer,
                    flags=re.DOTALL,
                )

                # Common Markdown compatibility.
                answer = re.sub(
                    r'(?<!\*)\*\*(.+?)\*\*(?!\*)',
                    r'<b>\1</b>',
                    answer,
                    flags=re.DOTALL,
                )
                answer = re.sub(
                    r'(?<!`)(`[^`\n]+`)(?!`)',
                    lambda m: f'<code>{m.group(1)[1:-1]}</code>',
                    answer,
                )

                # Markdown headings -> Rich HTML headings.
                answer = re.sub(r'(?m)^###\s+(.*)$', r'<h3>\1</h3>', answer)
                answer = re.sub(r'(?m)^##\s+(.*)$', r'<h2>\1</h2>', answer)
                answer = re.sub(r'(?m)^#\s+(.*)$', r'<h1>\1</h1>', answer)

                answer = answer.replace("\\n", "\n")

                # Telegram supports these named entities.
                answer = re.sub(
                    r'&(?!(?:amp|lt|gt|quot|apos|nbsp|hellip|mdash|ndash|lsquo|rsquo|ldquo|rdquo);)',
                    '&amp;',
                    answer,
                )

                # Preserve only the Rich HTML tags used by Sakura.
                allowed_tag_names = {
                    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
                    "code", "pre", "mark", "sub", "sup", "tg-spoiler",
                    "a", "p",
                    "h1", "h2", "h3", "h4", "h5", "h6",
                    "footer", "hr",
                    "ul", "ol", "li",
                    "blockquote", "aside", "cite",
                    "details", "summary",
                    "tg-math", "tg-math-block",
                    "tg-reference", "tg-emoji", "tg-time",
                    "table", "caption", "tr", "th", "td",
                }

                def sanitize_rich_tag(match: re.Match[str]) -> str:
                    raw = match.group(0)
                    closing = raw.startswith("</")
                    self_closing = raw.rstrip().endswith("/>")
                    name_match = re.match(r'</?\s*([a-zA-Z0-9-]+)', raw)

                    if not name_match:
                        return html.escape(raw)

                    tag = name_match.group(1).lower()

                    if tag not in allowed_tag_names:
                        return html.escape(raw)

                    # Closing tags never need attributes.
                    if closing:
                        return f"</{tag}>"

                    attrs = ""

                    # Extract and retain only known safe attributes.
                    raw_attrs = raw[name_match.end():]
                    if raw_attrs.endswith(">"):
                        raw_attrs = raw_attrs[:-1]
                    elif raw_attrs.endswith("/>"):
                        raw_attrs = raw_attrs[:-2]

                    attr_pattern = re.compile(
                        r'([a-zA-Z_:][a-zA-Z0-9_.:-]*)'
                        r'(?:\s*=\s*(".*?"|\'.*?\'|[^\s>]+))?'
                    )

                    safe_attrs = {
                        "a": {"href", "name"},
                        "pre": set(),
                        "code": {"class"},
                        "details": {"open"},
                        "ol": {"start", "type", "reversed"},
                        "li": {"value", "type"},
                        "td": {"colspan", "rowspan", "align", "valign"},
                        "th": {"colspan", "rowspan", "align", "valign"},
                        "tg-reference": {"name"},
                        "tg-emoji": {"emoji-id"},
                        "tg-time": {"unix", "format"},
                        "table": {"bordered", "striped"},
                    }

                    for attr_match in attr_pattern.finditer(raw_attrs):
                        attr_name = attr_match.group(1).lower()
                        attr_value = attr_match.group(2)

                        if attr_name not in safe_attrs.get(tag, set()):
                            continue

                        if attr_value is None:
                            attrs += f" {attr_name}"
                            continue

                        value = attr_value[1:-1] if attr_value[:1] in {'"', "'"} else attr_value

                        if tag == "a" and attr_name == "href":
                            if not (
                                value.startswith(("https://", "http://", "mailto:", "tel:", "tg://", "#"))
                            ):
                                continue

                        if tag == "code" and attr_name == "class":
                            if not re.fullmatch(r"language-[A-Za-z0-9_+-]+", value):
                                continue

                        if tag == "tg-time" and attr_name == "unix":
                            if not re.fullmatch(r"-?\d+", value):
                                continue

                        if tag in {"tg-emoji", "tg-reference"} and attr_name == "name":
                            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
                                continue

                        attrs += f' {attr_name}="{html.escape(value, quote=True)}"'

                    if self_closing:
                        return f"<{tag}{attrs}/>"
                    return f"<{tag}{attrs}>"

                # Sanitize tag syntax without destroying valid Rich HTML.
                answer = re.sub(
                    r'</?\s*[a-zA-Z0-9-]+(?:\s+[^<>]*?)?\s*/?>',
                    sanitize_rich_tag,
                    answer,
                )

                # Normalize excessive whitespace while keeping deliberate
                # Rich HTML structure intact.
                answer = re.sub(r'\n{4,}', '\n\n\n', answer).strip()

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