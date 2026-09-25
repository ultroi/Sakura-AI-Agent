# handlers/helpers.py
from __future__ import annotations

import html
import re

from telegram import Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import ContextTypes

MAX_MESSAGE_LENGTH = 4000  # Below Telegram's 4096 hard limit


def is_owner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Verifies that the incoming message originates from the pinned bot owner."""
    settings = context.application.bot_data["settings"]
    user = update.effective_user
    return bool(user and user.id == settings.owner_telegram_id)




# Telegram regular messages only support the basic HTML subset, while Sakura
# generates Rich Markdown. Rich Messages are the primary delivery path, but a
# regular-message fallback still needs to translate that Markdown instead of
# sending raw **bold** / # heading syntax with ParseMode.HTML.
_RICH_HTML_TAG_RE = re.compile(
    r"</?\s*(?:b|strong|i|em|u|ins|s|strike|del|span|tg-spoiler|a|tg-emoji|tg-time|code|pre|blockquote)(?:\s+[^<>]*?)?/?>",
    re.IGNORECASE,
)


def rich_markdown_to_html(text: str) -> str:
    """Convert Sakura's Rich Markdown into Telegram's regular-message HTML.

    This is only a fallback for clients/API paths where sendRichMessage is not
    available. Rich Messages remain the canonical formatting path.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"@@SAKURA_FMT_{len(placeholders) - 1}@@"

    # Protect fenced code before escaping and inline Markdown processing.
    def stash_fence(match: re.Match[str]) -> str:
        language = (match.group(1) or "").strip()
        body = html.escape(match.group(2), quote=False)
        lang_attr = f' class="language-{html.escape(language, quote=True)}"' if language else ""
        return stash(f"<pre><code{lang_attr}>{body}</code></pre>")

    text = re.sub(r"```(?:\s*([A-Za-z0-9_+.-]+))?\n([\s\S]*?)```", stash_fence, text)

    # Keep the small set of regular Telegram HTML tags that Sakura may emit
    # explicitly (for example voice transcription uses <i>...</i>).
    def stash_allowed_html(match: re.Match[str]) -> str:
        return stash(match.group(0))

    text = _RICH_HTML_TAG_RE.sub(stash_allowed_html, text)
    text = html.escape(text, quote=False)

    # Block-level Markdown -> readable regular-message HTML.
    lines = text.split("\n")
    converted: list[str] = []
    in_quote = False

    for line in lines:
        if re.match(r"^\s*&gt;", line):
            quote_line = re.sub(r"^\s*&gt;\s?", "", line)
            if not in_quote:
                converted.append("<blockquote>")
                in_quote = True
            converted.append(quote_line)
            continue

        if in_quote:
            converted.append("</blockquote>")
            in_quote = False

        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            converted.append(f"<b>{heading.group(1)}</b>")
            continue

        task = re.match(r"^\s*[-*+]\s+\[([ xX])\]\s+(.+)$", line)
        if task:
            mark = "☑" if task.group(1).lower() == "x" else "☐"
            converted.append(f"{mark} {task.group(2)}")
            continue

        unordered = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if unordered:
            converted.append(f"• {unordered.group(1)}")
            continue

        ordered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if ordered:
            converted.append(f"{ordered.group(1)}. {ordered.group(2)}")
            continue

        if re.match(r"^\s*(?:\*\s*){3,}$|^\s*(?:-\s*){3,}$|^\s*_{3,}\s*$", line):
            converted.append("────────────")
            continue

        converted.append(line)

    if in_quote:
        converted.append("</blockquote>")

    text = "\n".join(converted)

    # Inline Markdown. Order matters so links/code are protected before
    # emphasis markers can accidentally consume their punctuation.
    code_inline: list[str] = []

    def stash_inline_code(match: re.Match[str]) -> str:
        body = html.escape(match.group(1), quote=False)
        code_inline.append(f"<code>{body}</code>")
        return f"@@SAKURA_INLINE_CODE_{len(code_inline) - 1}@@"

    text = re.sub(r"`([^`\n]+)`", stash_inline_code, text)

    text = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^)\s]+|mailto:[^)\s]+|tel:[^)\s]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text, flags=re.DOTALL)
    text = re.sub(r"\|\|(.+?)\|\|", r"<tg-spoiler>\1</tg-spoiler>", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_", r"<i>\1</i>", text)

    for i, value in enumerate(code_inline):
        text = text.replace(f"@@SAKURA_INLINE_CODE_{i}@@", value)

    for i, value in enumerate(placeholders):
        text = text.replace(f"@@SAKURA_FMT_{i}@@", value)

    # Clean accidental excess blank lines introduced by conversion.
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()

def chunk_text(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Splits long text by line breaks or sentences to stay within Telegram limits."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_len:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            # If a single line exceeds max_len, force-slice it
            while len(line) > max_len:
                chunks.append(line[:max_len])
                line = line[max_len:]
            current_chunk.append(line)
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def should_sakura_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    message = update.message
    if not chat or not message:
        return False

    # Private chat me hamesha reply karegi
    if chat.type == ChatType.PRIVATE:
        return True

    # Group / Supergroup logic
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        # Agar message owner ka nahi hai, toh ignore karo
        if not is_owner(update, context):
            return False

        # Owner ka message hai, ab text/caption check karo ki "sakura" word hai ya nahi
        text = (message.text or message.caption or "").lower()
        
        # Check if "sakura" is in text or if bot's username is mentioned
        bot_username = context.bot.username.lower() if context.bot.username else "sakura"
        if "sakura" in text or f"@{bot_username}" in text:
            return True

    return False


async def send_response_safely(message_to_edit, full_text: str, chat):
    """Edits the placeholder message with the first chunk, and sends any overflow chunks as new messages."""
    # The normal answer contract is Rich Markdown. When Rich Messages are
    # unavailable, translate it to Telegram-compatible regular HTML first.
    rendered_text = rich_markdown_to_html(full_text)
    chunks = chunk_text(rendered_text)
    if not chunks:
        chunks = ["(Empty response)"]

    # Edit the initial thinking message with the first chunk
    await message_to_edit.edit_text(chunks[0], parse_mode=ParseMode.HTML)

    # Send any subsequent chunks as fresh messages
    for overflow_chunk in chunks[1:]:
        await chat.send_message(overflow_chunk, parse_mode=ParseMode.HTML)