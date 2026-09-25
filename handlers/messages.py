from __future__ import annotations
import asyncio
import contextlib
import html
import os
import secrets
import tempfile
from telegram import Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes
from handlers.helpers import (
    is_owner,
    rich_markdown_to_html,
    send_response_safely,
    should_sakura_reply,
)

# =============================================================================
# CONFIG
# =============================================================================

# Telegram Rich Messages support up to 32768 UTF-8 characters.
# We keep normal fallback messages below the Bot API's 4096-character limit.
MAX_NORMAL_MESSAGE_LENGTH = 4000
MAX_RICH_MESSAGE_LENGTH = 32768

THINKING_INTERVAL = 1.2


# =============================================================================
# SAFE FORMATTING HELPERS
# =============================================================================

async def safe_reply(message, text: str):
    """
    Send a message using Telegram HTML formatting.

    If Telegram rejects the HTML, retry as plain text.
    """
    try:
        return await message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
        )
    except BadRequest as exc:
        if "parse" in str(exc).lower():
            return await message.reply_text(text)
        raise


async def safe_edit(message, text: str):
    """
    Edit a message using Telegram HTML formatting.

    If Telegram rejects the HTML, retry as plain text.
    """
    try:
        await message.edit_text(
            text,
            parse_mode=ParseMode.HTML,
        )
    except BadRequest as exc:
        if "parse" in str(exc).lower():
            await message.edit_text(text)
        else:
            # Ignore harmless "message is not modified" style errors.
            pass


def split_text_safely(
    text: str,
    max_length: int = MAX_NORMAL_MESSAGE_LENGTH,
) -> list[str]:
    if not text:
        return [""]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        chunk = remaining[:max_length]

        split_at = chunk.rfind("\n")
        if split_at < max_length // 2:
            split_at = chunk.rfind(" ")

        if split_at <= 0:
            split_at = max_length

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


# =============================================================================
# TELEGRAM RICH MESSAGE / NATIVE AI THINKING
# =============================================================================

async def send_rich_message_draft(
    bot,
    chat_id: int,
    draft_id: int,
    text: str,
):
    safe_text = html.escape(text, quote=False)

    await bot.do_api_request(
        "sendRichMessageDraft",
        api_kwargs={
            "chat_id": chat_id,
            "draft_id": draft_id,
            "rich_message": {
                "html": f"<tg-thinking>{safe_text}</tg-thinking>",
            },
        },
    )


async def animate_native_thinking(
    bot,
    chat_id: int,
    draft_id: int,
    stages: tuple[str, ...],
):
    """
    Continuously update a native Telegram thinking draft.

    Changes using the same draft_id are animated by Telegram itself.
    """
    index = 0

    try:
        while True:
            await send_rich_message_draft(
                bot=bot,
                chat_id=chat_id,
                draft_id=draft_id,
                text=stages[index % len(stages)],
            )

            index += 1
            await asyncio.sleep(THINKING_INTERVAL)

    except asyncio.CancelledError:
        raise

    except Exception:
        # If Telegram temporarily rejects a draft update, stop the animation
        # quietly. The main request must still be allowed to finish.
        return


async def animate_legacy_thinking(
    message,
    stages: tuple[str, ...],
):
    index = 0

    try:
        while True:
            stage = stages[index % len(stages)]

            frame = (
                f"<blockquote>💭 "
                f"{html.escape(stage)}"
                f"</blockquote>"
            )

            await safe_edit(message, frame)

            index += 1
            await asyncio.sleep(THINKING_INTERVAL)

    except asyncio.CancelledError:
        raise

    except Exception:
        return


async def start_thinking(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    stages: tuple[str, ...],
):
    """
    Start Telegram's native thinking draft in private chats.

    Returns:
        (animation_task, thinking_message, draft_id)

    Native mode:
        animation_task != None
        thinking_message = None
        draft_id = non-zero int

    Legacy fallback:
        animation_task != None
        thinking_message = Telegram Message
        draft_id = None
    """
    chat = update.effective_chat
    bot = context.bot

    # sendRichMessageDraft is currently defined by Telegram for private chats.
    if chat and chat.type == ChatType.PRIVATE:
        draft_id = secrets.randbits(31) or 1

        try:
            # Make the first frame immediately so there is no visual delay.
            await send_rich_message_draft(
                bot=bot,
                chat_id=chat.id,
                draft_id=draft_id,
                text=stages[0],
            )

            task = asyncio.create_task(
                animate_native_thinking(
                    bot=bot,
                    chat_id=chat.id,
                    draft_id=draft_id,
                    stages=stages,
                )
            )

            return task, None, draft_id

        except TelegramError:
            pass

    # Legacy fallback.
    thinking_message = await update.effective_message.reply_text(
        f"<blockquote>💭 {html.escape(stages[0])}</blockquote>",
        parse_mode=ParseMode.HTML,
    )

    task = asyncio.create_task(
        animate_legacy_thinking(
            thinking_message,
            stages,
        )
    )

    return task, thinking_message, None


async def stop_thinking(task):
    """
    Cancel a thinking animation safely.
    """
    if task is None:
        return

    task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await task


# =============================================================================
# FINAL RESPONSE DELIVERY
# =============================================================================

async def send_rich_final_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    answer: str,
):
    """
    Persist the final answer with Telegram Rich Messages in any supported chat.

    The thinking draft itself is ephemeral, so Telegram expects the completed
    response to be sent with sendRichMessage.
    """
    chat = update.effective_chat
    source_message = update.effective_message

    if not chat:
        raise TelegramError("No effective chat is available for the response.")

    if not answer:
        answer = "I couldn't generate a response."

    # Rich Message limit from the current Bot API.
    if len(answer) > MAX_RICH_MESSAGE_LENGTH:
        raise TelegramError(
            "The generated response is longer than Telegram's "
            "Rich Message limit."
        )

    # Exactly one of html, markdown, or blocks is required by
    return await context.bot.do_api_request(
        "sendRichMessage",
        api_kwargs={
            "chat_id": chat.id,
            "rich_message": {
                "markdown": answer,
            },
            "reply_parameters": {
                "message_id": source_message.message_id,
            },
        },
    )


async def deliver_final_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    answer: str,
    thinking_message=None,
):
    chat = update.effective_chat

    # -------------------------------------------------------------------------
    # Canonical Rich Message path (private chats, groups, supergroups, and channels
    # where the bot is allowed to post). Telegram Rich Messages support advanced
    # Markdown including headings, lists, tables, quotes, and fenced code.
    # -------------------------------------------------------------------------
    try:
        await send_rich_final_response(
            update=update,
            context=context,
            answer=answer,
        )

        # Legacy thinking placeholders are only visual scaffolding. Once the
        # durable Rich Message is sent, remove the placeholder instead of
        # leaving a stale "Thinking..." message in the chat.
        if thinking_message is not None:
            with contextlib.suppress(TelegramError):
                await thinking_message.delete()
        return

    except TelegramError:
        # Rich Messages can fail on older/unsupported bot API paths. Fall back
        # to a regular Telegram message, but convert Sakura's Rich Markdown to
        # compatible HTML so formatting is not shown literally.
        pass

    if thinking_message is not None:
        await send_response_safely(
            thinking_message,
            answer,
            chat,
        )
        return

    rendered = rich_markdown_to_html(answer)
    chunks = split_text_safely(rendered)

    for index, chunk in enumerate(chunks):
        if index == 0:
            await safe_reply(
                update.effective_message,
                chunk,
            )
        else:
            await update.effective_message.chat.send_message(
                chunk,
                parse_mode=ParseMode.HTML,
            )





# =============================================================================
# MESSAGE HANDLERS
# =============================================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Handles standard text, media uploads, and replied-message context
    for Sakura's memory/vision.
    """
    if not update.message:
        return

    if not should_sakura_reply(update, context):
        return

    agent = context.application.bot_data["agent"]
    chat = update.effective_chat

    agent.current_message = update.effective_message

    user_text = (
        update.message.text
        or update.message.caption
        or ""
    )
    reply_info = ""

    if update.message.reply_to_message:
        replied_msg = update.message.reply_to_message

        replied_text = (
            replied_msg.text
            or replied_msg.caption
            or ""
        )

        sender_name = (
            replied_msg.from_user.first_name
            if replied_msg.from_user
            else "Someone"
        )

        if replied_msg.photo:
            rep_file_id = replied_msg.photo[-1].file_id

            reply_info = (
                f"[Context: Senpai is replying to a PHOTO sent by "
                f"{sender_name} (file_id: {rep_file_id}) with text: "
                f"'{replied_text}']\n"
            )

        elif replied_msg.document:
            rep_file_id = replied_msg.document.file_id
            rep_name = (
                replied_msg.document.file_name
                or "document.pdf"
            )

            reply_info = (
                f"[Context: Senpai is replying to a DOCUMENT sent by "
                f"{sender_name} (file_id: {rep_file_id} "
                f"file_name: {rep_name}) with text: "
                f"'{replied_text}']\n"
            )

        elif replied_msg.video:
            rep_file_id = replied_msg.video.file_id

            reply_info = (
                f"[Context: Senpai is replying to a VIDEO sent by "
                f"{sender_name} (file_id: {rep_file_id}) with text: "
                f"'{replied_text}']\n"
            )

        elif replied_msg.voice:
            rep_file_id = replied_msg.voice.file_id

            reply_info = (
                f"[Context: Senpai is replying to a VOICE NOTE sent by "
                f"{sender_name} (file_id: {rep_file_id})]\n"
            )

        elif replied_text:
            reply_info = (
                f"[Context: Senpai is replying to {sender_name}'s "
                f"message: '{replied_text}']\n"
            )

    media_info = ""

    if update.message.photo:
        file_id = update.message.photo[-1].file_id

        media_info = (
            f"[System: Senpai sent a PHOTO. file_id: {file_id}]\n"
        )

    elif update.message.video:
        file_id = update.message.video.file_id

        media_info = (
            f"[System: Senpai sent a VIDEO. file_id: {file_id}]\n"
        )

    elif update.message.document:
        file_id = update.message.document.file_id
        file_name = (
            update.message.document.file_name
            or "document.pdf"
        )

        media_info = (
            f"[System: Senpai sent a DOCUMENT. "
            f"file_id: {file_id} file_name: {file_name}]\n"
        )

    elif update.message.voice:
        file_id = update.message.voice.file_id

        media_info = (
            f"[System: Senpai sent a VOICE NOTE. file_id: {file_id}]\n"
        )

    # -------------------------------------------------------------------------
    # Combine reply context, media info and user text
    # -------------------------------------------------------------------------
    final_prompt = reply_info + media_info + user_text

    if not final_prompt.strip():
        final_prompt = (
            "[System: User sent media with no caption.]"
        )

    # -------------------------------------------------------------------------
    # Start native Telegram thinking
    # -------------------------------------------------------------------------
    await chat.send_action(ChatAction.TYPING)

    thinking_stages = (
        "Thinking",

    )

    animation_task, thinking_msg, _draft_id = await start_thinking(
        update=update,
        context=context,
        stages=thinking_stages,
    )

    try:
        answer = await agent.respond(
            telegram_id=update.effective_user.id,
            chat_id=chat.id,
            user_text=final_prompt,
        )

    except Exception as exc:
        answer = (
            f"I ran into an internal issue: {exc}"
        )

    finally:
        await stop_thinking(animation_task)

    # -------------------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------------------
    await deliver_final_response(
        update=update,
        context=context,
        answer=answer,
        thinking_message=thinking_msg,
    )


# =============================================================================
# VOICE HANDLER
# =============================================================================

async def voice_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Handles audio processing and transcription via Groq/Whisper.
    """
    if (
        not update.message
        or not update.message.voice
    ):
        return

    if not is_owner(update, context):
        await update.message.reply_text(
            "🌸 This is a private assistant.",
        )
        return

    agent = context.application.bot_data["agent"]

    agent.telegram_id = update.effective_user.id
    agent.chat_id = update.effective_chat.id

    # -------------------------------------------------------------------------
    # Cache current message for voice-related metadata
    # -------------------------------------------------------------------------
    agent.current_message = update.effective_message

    chat = update.effective_chat

    await chat.send_action(ChatAction.TYPING)

    # -------------------------------------------------------------------------
    # Start native/fallback thinking
    # -------------------------------------------------------------------------
    transcription_stages = (
        "Listening to your voice message",
        "Transcribing the audio",
        "Understanding the transcription",
    )

    animation_task, thinking_msg, draft_id = await start_thinking(
        update=update,
        context=context,
        stages=transcription_stages,
    )

    temp_path = None

    try:
        # ---------------------------------------------------------------------
        # Download voice file
        # ---------------------------------------------------------------------
        voice = await update.message.voice.get_file()

        with tempfile.NamedTemporaryFile(
            suffix=".ogg",
            delete=False,
        ) as tmp:
            temp_path = tmp.name

        await voice.download_to_drive(temp_path)

        with open(temp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        # ---------------------------------------------------------------------
        # Transcribe using Groq Whisper
        # ---------------------------------------------------------------------
        transcription = await asyncio.to_thread(
            agent.groq.audio.transcriptions.create,
            file=("voice.ogg", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="json",
        )

        text = getattr(
            transcription,
            "text",
            "",
        ) or ""

        # ---------------------------------------------------------------------
        # Empty transcription
        # ---------------------------------------------------------------------
        if not text.strip():
            await stop_thinking(animation_task)

            if thinking_msg is not None:
                await safe_edit(
                    thinking_msg,
                    (
                        "I couldn't hear anything clearly "
                        "in that voice message."
                    ),
                )
            else:
                await update.message.reply_text(
                    "I couldn't hear anything clearly "
                    "in that voice message."
                )

            return

        # ---------------------------------------------------------------------
        # Switch thinking stage from transcription to answer generation
        # ---------------------------------------------------------------------
        await stop_thinking(animation_task)

        answer_stages = (
            "Thinking",
            "Thinking.",
            "Thinking..",
            "Thinking...",
            "Thinking..",
            "Thinking.",
            "Thinking..",
            "Thinking...",
            "Thinking..",
            "Thinking.",
            "Thinking",
            
        )

        # ---------------------------------------------------------------------
        # Native Rich Message mode
        # ---------------------------------------------------------------------
        if draft_id is not None:
            animation_task = asyncio.create_task(
                animate_native_thinking(
                    bot=context.bot,
                    chat_id=chat.id,
                    draft_id=draft_id,
                    stages=answer_stages,
                )
            )

            # Immediately show the next frame.
            with contextlib.suppress(TelegramError):
                await send_rich_message_draft(
                    bot=context.bot,
                    chat_id=chat.id,
                    draft_id=draft_id,
                    text=answer_stages[0],
                )

        # ---------------------------------------------------------------------
        # Legacy fallback mode
        # ---------------------------------------------------------------------
        elif thinking_msg is not None:
            animation_task = asyncio.create_task(
                animate_legacy_thinking(
                    thinking_msg,
                    answer_stages,
                )
            )

        # ---------------------------------------------------------------------
        # Ask Sakura for the final answer
        # ---------------------------------------------------------------------
        answer = await agent.respond(
            update.effective_user.id,
            update.effective_chat.id,
            text,
        )

        # Safely include transcription in the final response.
        escaped_transcription = html.escape(text)

        final_text = (
            f"🎙️ <i>{escaped_transcription}</i>\n\n"
            f"{answer}"
        )
        await stop_thinking(animation_task)


        if draft_id is not None and thinking_msg is None:
            try:
                await send_rich_final_response(
                    update=update,
                    context=context,
                    answer=final_text,
                )

            except TelegramError:
                chunks = split_text_safely(rich_markdown_to_html(final_text))

                for index, chunk in enumerate(chunks):
                    if index == 0:
                        await safe_reply(
                            update.message,
                            chunk,
                        )
                    else:
                        await update.message.chat.send_message(
                            chunk,
                            parse_mode=ParseMode.HTML,
                        )

        else:
            remaining = rich_markdown_to_html(final_text)
            is_first_chunk = True

            while remaining:
                chunk = remaining[:MAX_NORMAL_MESSAGE_LENGTH]

                if len(remaining) > MAX_NORMAL_MESSAGE_LENGTH:
                    split_at = chunk.rfind("\n")

                    if (
                        split_at
                        < MAX_NORMAL_MESSAGE_LENGTH // 2
                    ):
                        split_at = chunk.rfind(" ")

                    if split_at > 0:
                        chunk = chunk[:split_at]

                # First chunk replaces the thinking placeholder.
                if is_first_chunk:
                    await safe_edit(
                        thinking_msg,
                        chunk,
                    )
                    is_first_chunk = False

                else:
                    await safe_reply(
                        update.message,
                        chunk,
                    )

                remaining = remaining[len(chunk):].lstrip()

    except Exception as exc:
        await stop_thinking(animation_task)

        error_text = (
            "Voice processing error: "
            f"<code>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc))}</code>"
        )

        if thinking_msg is not None:
            await safe_edit(
                thinking_msg,
                error_text,
            )
        else:
            await safe_reply(
                update.message,
                error_text,
            )

    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass