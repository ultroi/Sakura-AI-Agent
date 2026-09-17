# handlers/helpers.py
from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

MAX_MESSAGE_LENGTH = 4000  # Below Telegram's 4096 hard limit


def is_owner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Verifies that the incoming message originates from the pinned bot owner."""
    settings = context.application.bot_data["settings"]
    user = update.effective_user
    return bool(user and user.id == settings.owner_telegram_id)


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


async def send_response_safely(message_to_edit, full_text: str, chat):
    """Edits the placeholder message with the first chunk, and sends any overflow chunks as new messages."""
    chunks = chunk_text(full_text)
    if not chunks:
        chunks = ["(Empty response)"]

    # Edit the initial thinking message with the first chunk
    await message_to_edit.edit_text(chunks[0], parse_mode=ParseMode.HTML)

    # Send any subsequent chunks as fresh messages
    for overflow_chunk in chunks[1:]:
        await chat.send_message(overflow_chunk, parse_mode=ParseMode.HTML)