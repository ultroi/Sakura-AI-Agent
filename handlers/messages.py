from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
import contextlib
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from handlers.helpers import is_owner, send_response_safely


# --- HELPER FUNCTIONS FOR SAFE FORMATTING ---

async def safe_reply(message, text: str):
    """Tries to send a new message with HTML parsing. Falls back to plain text."""
    try:
        return await message.reply_text(text, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "parse" in str(e).lower():
            return await message.reply_text(text)
        else:
            raise

async def safe_edit(message, text: str):
    """Tries to edit an existing message with HTML parsing. Falls back to plain text."""
    try:
        await message.edit_text(text, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "parse" in str(e).lower():
            await message.edit_text(text)
        else:
            pass # Ignore if message is exactly the same


async def animate_thinking(message, base_text="Sakura is thinking"):
    """
    Background task that updates a message to create an animation.
    Uses Telegram's <blockquote> tag for a beautiful UI feel.
    """
    frames = [
        f"<blockquote>💭 {base_text}.</blockquote>",
        f"<blockquote>💭 {base_text}..</blockquote>",
        f"<blockquote>💭 {base_text}...</blockquote>",
        f"<blockquote>✨ {base_text}.</blockquote>",
        f"<blockquote>✨ {base_text}..</blockquote>",
        f"<blockquote>✨ {base_text}...</blockquote>",
    ]
    idx = 0
    while True:
        try:
            # Wait first to avoid instant rate limit hits on Telegram's API
            await asyncio.sleep(1.2)
            await message.edit_text(frames[idx % len(frames)], parse_mode=ParseMode.HTML)
            idx += 1
        except asyncio.CancelledError:
            # Stop the loop when the main task cancels it
            break
        except Exception:
            await asyncio.sleep(1)


# --- MESSAGE HANDLERS ---

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles standard text as well as intercepting media uploads for Sakura's memory/vision."""
    if not is_owner(update, context):
        await update.message.reply_text("🌸 This is a private assistant.")
        return

    agent = context.application.bot_data["agent"]
    chat = update.effective_chat
    
    # Extract text or caption
    user_text = update.message.text or update.message.caption or ""
    media_info = ""
    
    # Check if the message contains any media and extract its file_id
    if update.message.photo:
        file_id = update.message.photo[-1].file_id # Best resolution
        media_info = f"[System: Senpai sent a PHOTO. file_id: {file_id}]\n"
    elif update.message.video:
        file_id = update.message.video.file_id
        media_info = f"[System: Senpai sent a VIDEO. file_id: {file_id}]\n"
    elif update.message.document:
        file_id = update.message.document.file_id
        media_info = f"[System: Senpai sent a DOCUMENT. file_id: {file_id}]\n"
    elif update.message.voice:
        file_id = update.message.voice.file_id
        media_info = f"[System: Senpai sent a VOICE NOTE. file_id: {file_id}]\n"

    # Combine media info with whatever text the user typed
    final_prompt = media_info + user_text
    if not final_prompt.strip():
        final_prompt = "[System: User sent media with no caption.]"

    # Send typing action
    await chat.send_action(ChatAction.TYPING)
    thinking_msg = await update.message.reply_text("<i>Sakura is thinking...</i>", parse_mode="HTML")

    animation_task = asyncio.create_task(animate_thinking(thinking_msg))

    try:
        answer = await agent.respond(
            telegram_id=update.effective_user.id,
            chat_id=chat.id,
            user_text=final_prompt
        )
    except Exception as exc:
        answer = f"I ran into an internal issue: {exc}"
    finally:
        # Await task cancellation safely to eliminate race conditions
        animation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await animation_task

    # Chunk-safe reply delivery
    await send_response_safely(thinking_msg, answer, chat)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles audio processing and transcription via Groq/Whisper."""
    if not update.message or not update.message.voice:
        return
    
    agent = context.application.bot_data["agent"]
    agent.telegram_id = update.effective_user.id
    agent.chat_id = update.effective_chat.id
    
    await update.message.chat.send_action("typing")

    # Initial placeholder for voice
    thinking_msg = await update.message.reply_text(
        "<blockquote>🎙️ Listening and transcribing...</blockquote>", 
        parse_mode=ParseMode.HTML
    )
    # Start animation with a custom audio message
    animation_task = asyncio.create_task(animate_thinking(thinking_msg, "Processing audio"))

    temp_path = None
    try:
        voice = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            temp_path = tmp.name
        
        await voice.download_to_drive(temp_path)
        
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        transcription = await asyncio.to_thread(
            agent.groq.audio.transcriptions.create,
            file=("voice.ogg", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="json",
        )
        
        text = getattr(transcription, "text", "") or ""
        
        if not text.strip():
            animation_task.cancel()
            await safe_edit(thinking_msg, "I couldn't hear anything clearly in that voice message.")
            return
            
        # Voice is transcribed, let the user know we are generating the answer now
        answer = await agent.respond(
            update.effective_user.id,
            update.effective_chat.id,
            text,
        )

        final_text = f"🎙️ <i>{text}</i>\n\n{answer}"

        # Stop animation
        animation_task.cancel()

        MAX_LENGTH = 4000
        remaining = final_text
        is_first_chunk = True

        while remaining:
            chunk = remaining[:MAX_LENGTH]

            if len(remaining) > MAX_LENGTH:
                split_at = chunk.rfind("\n")
                if split_at < MAX_LENGTH // 2:
                    split_at = chunk.rfind(" ")
                if split_at > 0:
                    chunk = chunk[:split_at]

            # Replace the thinking message with the first chunk, send the rest as new messages
            if is_first_chunk:
                await safe_edit(thinking_msg, chunk)
                is_first_chunk = False
            else:
                await safe_reply(update.message, chunk)
                
            remaining = remaining[len(chunk):].lstrip()
            
    except Exception as exc:
        animation_task.cancel()
        await safe_edit(thinking_msg, f"Voice processing error: <code>{type(exc).__name__}: {exc}</code>")
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass