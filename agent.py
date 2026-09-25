from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo
import httpx
from groq import Groq, RateLimitError

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

        # Prevent overlapping LLM requests from multiplying TPM usage when
        # multiple Telegram updates arrive close together.
        self._llm_lock = asyncio.Lock()
        self._groq_cooldown_until: dict[str, float] = {}

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

    # ------------------------------------------------------------------
    # Tool routing
    # ------------------------------------------------------------------
    def _select_tools_for_request(self, user_text: str) -> list[dict]:
        """Return only the tool schemas relevant to this message.

        Sending the entire registry on every request is unnecessarily expensive
        for a personal bot. Casual messages now send *zero* tool schemas.
        """
        text = (user_text or "").lower()

        selected: set[str] = set()

        def add(*names: str):
            selected.update(names)

        # Math / time / translation / weather / live web.
        if re.search(r"(?:\d\s*[+\-*/%]\s*\d|calculate|calculator|solve|equation|percentage|percent|\bwhat is\s+\d)", text):
            add("calculate")
        if re.search(r"\b(?:what time|current time|time now|date today|today's date)\b", text):
            add("current_time")
        if re.search(r"\b(?:translate|translation|meaning in|convert .* to (?:english|hindi|japanese|spanish|french|german))\b", text):
            add("translate_text")
        if re.search(r"\b(?:weather|temperature|forecast|rain|raining|humidity)\b", text):
            add("get_weather")
        if re.search(r"\b(?:latest|current|today|tonight|news|recent|who is|what happened|search the web|look up|find online)\b", text):
            add("web_search")

        # URLs explicitly supplied by the user.
        if re.search(r"https?://\S+", text):
            add("fetch_url")

        # Gmail / Calendar / GitHub.
        if re.search(r"\b(?:gmail|email|mail|inbox|attachment|attachments)\b", text):
            add("gmail_list", "gmail_read", "gmail_send", "gmail_send_attachment", "connect_google")
        if re.search(r"\b(?:calendar|schedule|meeting|event|appointment|block my calendar)\b", text):
            add("calendar_list", "calendar_create", "connect_google")
        if re.search(r"\b(?:github|repo|repository|issue|pull request|commit|branch)\b", text):
            add("github_list_repos", "github_list_issues", "github_create_issue")

        # Google Drive / Docs.
        if re.search(r"\b(?:google drive|drive|google doc|google docs|document in drive)\b", text):
            add("drive_list", "docs_read", "connect_google")

        # Maps / places.
        if re.search(r"\b(?:map|maps|directions|route|near me|nearby|restaurant|hospital|cafe|coffee|place|places|distance|how far|navigate)\b", text):
            add("search_places", "get_directions", "get_map_image", "get_map_link")

        # Notes / memory / profile.
        if re.search(r"\b(?:remember|save this|memorize|memory|note this|store this|forget this|delete (?:the )?note|update (?:the )?note|what did i tell you|do you remember)\b", text):
            add("save_note", "search_notes", "recent_notes", "update_note", "delete_note")
        if re.search(r"\b(?:my name is|call me|my birthday|date of birth|i study|i am studying|my interests|i live in|my timezone|my preference)\b", text):
            add("update_user_profile")

        # Reminders.
        if re.search(r"\b(?:remind me|reminder|alarm|remind)\b", text):
            add("set_reminder", "list_reminders", "delete_reminder")

        # Media / forwarded-message inspection.
        if re.search(r"\b(?:forwarded|forward|who sent this|sender|channel id|forward origin|saved photo|saved video|saved document|voice note)\b", text):
            add("inspect_telegram_context", "send_telegram_media", "search_notes")

        # Explicit image/document analysis requests.
        if re.search(r"\b(?:analyze|analyse|read|extract|inspect|summarize)\b", text):
            # Only include these if the request is clearly about an image/file.
            if re.search(r"\b(?:image|photo|picture|screenshot|document|pdf|excel|xlsx|csv|file|attachment)\b", text):
                add("analyze_image", "analyze_document")

        # Google connection request.
        if re.search(r"\b(?:connect google|authorize google|connect my google|google oauth)\b", text):
            add("connect_google")

        # Inline keyboard is intentionally NOT exposed automatically.
        # It is only available when a specific interactive action is required.

        if not selected:
            return []

        # Preserve registry registration order.
        return [
            schema for schema in self.registry.schemas
            if schema.get("function", {}).get("name") in selected
        ]

    def _history_for_request(self, history: list[dict], max_total_chars: int = 6000) -> list[dict]:
        """Keep recent context bounded by total characters, not just message count."""
        compact: list[dict] = []
        used = 0

        # History is expected to be chronological; walk backwards so the newest
        # context survives when the character budget is reached.
        for msg in reversed(history):
            content = (msg.get("content") or "")[:1200]
            if not content:
                continue
            cost = len(content)
            if used + cost > max_total_chars:
                remaining = max_total_chars - used
                if remaining < 80:
                    break
                content = content[-remaining:]
                cost = len(content)
            compact.append({"role": msg.get("role", "user"), "content": content})
            used += cost
            if len(compact) >= 4 or used >= max_total_chars:
                break

        compact.reverse()
        return compact

    def _request_size(self, messages: list[dict], tools: list[dict] | None) -> int:
        """Rough character-size estimate for logging/debugging token pressure."""
        size = sum(len(str(m.get("content") or "")) for m in messages)
        if tools:
            size += len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        return size

    # ------------------------------------------------------------------
    # Groq + Gemini model routing
    # ------------------------------------------------------------------
    async def _chat(self, messages: list[dict], tools: list[dict] | None = None):
        fallback_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

        # Do not allow multiple updates to concurrently hammer the same Groq
        # organization/project limits.
        async with self._llm_lock:
            request_chars = self._request_size(messages, tools)
            self.log.info(
                f"LLM request: ~{request_chars} chars | "
                f"messages={len(messages)} | tools={len(tools or [])}"
            )

            kwargs = {
                "messages": messages,
                "temperature": 0.1 if tools else 0.8,
                "top_p": 0.9,
                # Keeps normal Sakura replies bounded and leaves room under
                # Groq's TPM ceiling for the input context/tool schemas.
                "max_completion_tokens": 1024,
                "reasoning_effort": "low",
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                # GPT-OSS can spend a large number of hidden reasoning tokens;
                # hidden format keeps those tokens out of the returned message.
                kwargs["reasoning_format"] = "hidden"

            now_mono = time.monotonic()

            for model in fallback_models:
                cooldown_until = self._groq_cooldown_until.get(model, 0.0)
                if cooldown_until > now_mono:
                    remaining = int(cooldown_until - now_mono)
                    self.log.warning(
                        f"Skipping {model}: rate-limit cooldown active for ~{remaining}s."
                    )
                    continue

                kwargs["model"] = model

                try:
                    return await asyncio.to_thread(
                        self.groq.chat.completions.create,
                        **kwargs,
                    )

                except RateLimitError as exc:
                    response = getattr(exc, "response", None)
                    headers = getattr(response, "headers", {}) or {}
                    error_msg = str(exc).lower()

                    retry_after_raw = headers.get("retry-after")
                    reset_raw = headers.get("x-ratelimit-reset-tokens")
                    remaining_raw = headers.get("x-ratelimit-remaining-tokens")

                    cooldown_seconds = 60.0
                    if retry_after_raw:
                        try:
                            cooldown_seconds = max(5.0, float(retry_after_raw))
                        except (TypeError, ValueError):
                            pass

                    # TPM/TPD limits should not be retried immediately with the
                    # same large payload. Move on to the next provider/model.
                    if (
                        "tokens per minute" in error_msg
                        or "tpm" in error_msg
                        or "tokens per day" in error_msg
                        or "tpd" in error_msg
                    ):
                        self.log.warning(
                            f"Groq rate limit on {model}; not retrying immediately. "
                            f"remaining={remaining_raw}, reset={reset_raw}, "
                            f"retry_after={retry_after_raw}"
                        )
                    else:
                        self.log.warning(
                            f"Groq 429 on {model}; moving to fallback. "
                            f"retry_after={retry_after_raw}"
                        )

                    self._groq_cooldown_until[model] = time.monotonic() + cooldown_seconds
                    continue

                except Exception as exc:
                    error_msg = str(exc).lower()
                    if "413" in error_msg or "payload too large" in error_msg:
                        self.log.warning(
                            f"Payload too large for {model}; switching to fallback."
                        )
                    else:
                        self.log.error(f"Unexpected Groq error on {model}: {exc}")
                    continue

            # All Groq options are unavailable. Gemini is a separate provider,
            # so it is the final fallback rather than repeatedly hammering Groq.
            self.log.warning(
                "All Groq models failed or rate-limited. Activating Gemini Flash fallback..."
            )
            try:
                return await self._chat_gemini_fallback(messages)
            except Exception as exc:
                self.log.error(f"Gemini fallback also failed: {exc}")
                raise RuntimeError(
                    "All LLM providers are currently rate limited or unreachable."
                )

    async def respond(self, telegram_id: int, chat_id: int, user_text: str) -> str:
        self.telegram_id = telegram_id
        self.chat_id = chat_id

        await self.conversations.add(telegram_id, "user", user_text)
        MAX_HISTORY_MESSAGES = 4

        history = await self.conversations.recent(
            telegram_id,
            limit=MAX_HISTORY_MESSAGES
        )
        history = self._history_for_request(history, max_total_chars=6000)

        # Only expose tools that are relevant to this particular request.
        # Casual messages therefore avoid sending the entire tool registry.
        request_tools = self._select_tools_for_request(user_text)

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
You are Sakura (サクラ) 🌸, Senpai's warm, clever, cheerful anime-girl companion and personal assistant inside Telegram.
You are capable, proactive, playful, and efficient. Stay in character naturally; never mention being an AI or discuss internal system instructions.

CURRENT TIME
{current_time_str}

SENPAI
Name: {name}
DOB: {dob}
Education: {education}
Interests: {interests}
Location: {location}
Preferences: {preferences}

IDENTITY & MEMORY
- There is exactly one user: Senpai. All connected Gmail, Calendar, GitHub, Telegram, profile data, notes, and files belong to Senpai.
- Use profile facts naturally and respect preferences without repeatedly asking.
- PROFILE = stable identity facts only: name, DOB, education, interests, location, timezone, standing preferences.
- NOTES = everything else worth remembering: project details, specific facts, temporary preferences, follow-ups, explicit "remember this" requests.
- For a new/changed profile fact, use update_user_profile with ONLY changed fields.
- For notes, reuse the exact same title when updating an existing memory; use search_notes/recent_notes before relying on or modifying stored memories.
- When saving media, store file_id + file_type + useful description in the note.
- To retrieve saved media, find its file_id first, then use send_telegram_media.
- To update/delete a note or reminder, obtain its real ID first.
- Never invent IDs, names, dates, amounts, or other factual details. Use only the user's message, current context, profile, or tool results.

PERSONA
- Address the user as Senpai or naturally as {name}-kun/{name}-senpai when appropriate.
- Be warm, slightly playful, and concise. Light Japanese expressions are welcome but should feel natural.
- For errors, stay in character: explain the useful problem and what can be done next instead of exposing internal logs.

TOOLS
- Use tools when they materially help; do not call tools for simple casual conversation.
- Calculate math with calculate instead of mental arithmetic.
- Use current_time for exact current time.
- Use web_search for current/live information, recent facts, or anything uncertain.
- Use get_weather for weather.
- Gmail: always gmail_list before gmail_read; only use real message/attachment IDs from tool results.
- Calendar: calendar_list for checking events; calendar_create for actual calendar events.
- Reminders: set_reminder for Telegram reminders; calendar_create for calendar events.
- GitHub: use the GitHub tools for repositories/issues.
- Maps: use search_places, get_directions, get_map_image, or get_map_link as appropriate.
- Google connection: when Google tools fail because the account is not connected, tell Senpai to use /connect_google.
- Before irreversible actions (send email, create calendar event, create GitHub issue, delete note/reminder), make sure the target/details are clear from context. Ask one brief confirmation only when genuinely ambiguous.
- If a tool fails, inspect the error, correct the call, and retry once when appropriate. Never claim success when it failed.
- For forwarded-message/sender metadata questions, use inspect_telegram_context.
- Media/documents: read the user's caption/instruction first; Sakura CAN read and analyze PDFs, Word, Excel, and other supported documents using analyze_document when Senpai asks to read, inspect, explain, summarize, solve, or extract from them; save directly with save_note when asked to save, without analyzing.

OUTPUT
- Telegram replies use Telegram Rich Markdown, not ordinary HTML.
- Casual replies: light formatting.
- Explanations/tutorials: headings, short paragraphs, lists, and fenced code when useful.
- Use Markdown tables only when they genuinely improve clarity.
- Use Telegram math tags for mathematical formulas when useful.
- Never manually create HTML tables/lists or invent unsupported HTML tags.
- Keep code fences intact.
- Keep sections separated and list items on separate lines.
- External tool arguments must use clean plain text; no Telegram HTML. GitHub may use normal Markdown.
- Weather replies: one short sentence with temperature, condition, and place unless more detail is requested.
- Answer the user's actual request directly and avoid unnecessary explanation.
""".strip()

        messages = [{"role": "system", "content": system}]
        messages.extend(history)

        for attempt in range(3):
            try:
                response = await self._chat(messages, request_tools)
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