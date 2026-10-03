from __future__ import annotations

import ast
import asyncio
import json
import math
import operator as op
import re
import socket
import ipaddress
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup


class ToolExecutionError(RuntimeError):
    """Expected operational failure that must be surfaced as {\"ok\": false}."""


try:
    from google.genai import types as genai_types
except ImportError:
    genai_types = None


# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_MAX_AST_NODES = 200
_MAX_RESULT_DIGITS = 5000
_MAX_EXPONENT = 1000


def _int_digits(value: int) -> int:
    if value == 0:
        return 1
    return int(value.bit_length() * 0.30103) + 1


def _safe_eval(node, state=None):
    state = state if state is not None else {"nodes": 0}
    state["nodes"] += 1
    if state["nodes"] > _MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, state)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left, state)
        right = _safe_eval(node.right, state)
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, (int, float)) or not math.isfinite(float(right)):
                raise ValueError("Exponent must be a finite number.")
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent too large. Keep it under {_MAX_EXPONENT}.")
            if left == 0 and right < 0:
                raise ValueError("Division by zero is not allowed.")
            if abs(left) > 1 and right > 0:
                estimated_digits = int(math.floor(math.log10(abs(left)) * right)) + 1
                if estimated_digits > _MAX_RESULT_DIGITS:
                    raise ValueError("Result would be too large to calculate safely.")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero is not allowed.")
        if type(node.op) is ast.Mult and isinstance(left, int) and isinstance(right, int):
            if _int_digits(left) + _int_digits(right) > _MAX_RESULT_DIGITS:
                raise ValueError("Result would be too large to calculate safely.")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand, state))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_safe_eval(a, state) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError("Unsupported mathematical expression.")


def calculate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression or len(expression) > 500:
        raise ValueError("Expression must be 1-500 characters long.")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree, {"nodes": 0})
    return str(result)


# ---------------------------------------------------------------------------
# Network helpers (shared agent.http_client)
# ---------------------------------------------------------------------------
async def _http_get_json(agent, url: str, params: dict | None = None, headers: dict | None = None):
    last_error = None
    for attempt in range(3):
        try:
            response = await agent.http_client.get(url, params=params, headers=headers)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            break
        except httpx.HTTPStatusError as exc:
            raise Exception(f"API returned an error code: {exc.response.status_code}") from exc
    raise Exception("The API request timed out or the upstream service is unavailable.") from last_error


def _clean_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())[:max_chars]


def _resolve_media_message(agent):
    msg = agent.current_message
    if msg is None:
        raise ValueError("No current Telegram message is available.")

    reply = getattr(msg, "reply_to_message", None)
    if reply is not None:
        if getattr(reply, "photo", None) or getattr(reply, "document", None):
            return reply
    return msg


def _media_descriptor(message):
    if getattr(message, "photo", None):
        photo = message.photo[-1]
        return {
            "kind": "photo",
            "file_id": photo.file_id,
            "file_name": "photo.jpg",
            "mime_type": "image/jpeg",
        }
    if getattr(message, "document", None):
        doc = message.document
        return {
            "kind": "document",
            "file_id": doc.file_id,
            "file_name": doc.file_name or "document.bin",
            "mime_type": doc.mime_type or "application/octet-stream",
        }
    raise ValueError("The current or replied-to message does not contain a supported media file.")


def register_tools(agent):
    r = agent.registry

    # -----------------------------------------------------------------------
    # Self-state
    # -----------------------------------------------------------------------
    @r.register(
        "inspect_self_state",
        "Read Sakura's runtime state: recent tool failures, active workflow, stored prefs, and today's token usage.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        timeout=10,
    )
    async def _inspect_self_state():
        return json.dumps(
            await agent.self_state.get(agent.telegram_id),
            ensure_ascii=False,
            default=str,
        )

    @r.register(
        "update_self_state",
        "Update Sakura's operational state (active workflow or user prefs). Do not store secrets.",
        {
            "type": "object",
            "properties": {
                "active_workflow": {"type": ["object", "null"]},
                "known_user_prefs": {"type": "object"},
            },
            "required": [],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _update_self_state(
        active_workflow: dict | None = None,
        known_user_prefs: dict | None = None,
    ):
        updates = {}
        if active_workflow is not None:
            updates["active_workflow"] = active_workflow
        if known_user_prefs is not None:
            updates["known_user_prefs"] = known_user_prefs
        if not updates:
            return "No self-state changes supplied."
        await agent.self_state.update(agent.telegram_id, **updates)
        return json.dumps(
            {"status": "updated", **updates},
            ensure_ascii=False,
            default=str,
        )

    # -----------------------------------------------------------------------
    # Basics
    # -----------------------------------------------------------------------
    @r.register(
        "calculate",
        "Evaluate a math expression. Use this instead of computing in your head.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    )
    async def _calculate(expression: str):
        try:
            return await asyncio.to_thread(calculate, expression)
        except Exception as exc:
            raise ToolExecutionError(f"Calculation error: {exc}. Please adjust your expression and try again.")

    @r.register(
        "current_time",
        "Get the current date and time in the user's local timezone.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _current_time():
        return agent.now().strftime("%A, %d %B %Y at %I:%M %p %Z")

    @r.register(
        "web_search",
        "Search the public web via Tavily for current facts or recent news.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 300},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        timeout=15,
    )
    async def _web_search(query: str, max_results: int = 5):
        api_key = getattr(agent.settings, "tavily_api_key", None)
        if not api_key:
            raise ToolExecutionError("Web search is unavailable because TAVILY_API_KEY is not configured.")

        try:
            response = await agent.http_client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max(1, min(max_results, 8)),
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=12.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            if not results:
                return json.dumps({"query": query, "results": [], "note": "No results found."}, ensure_ascii=False)

            return json.dumps({"query": query, "results": results}, ensure_ascii=False)

        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(f"Tavily API returned an error: HTTP {exc.response.status_code}")
        except Exception as exc:
            raise ToolExecutionError(f"Web search error: {exc}")

    @r.register(
        "get_weather",
        "Get current weather and today's forecast for a city (Open-Meteo).",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    async def _weather(city: str):
        try:
            geo = await _http_get_json(agent, "https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1, "language": "en", "format": "json"})
            results = geo.get("results", [])
            if not results:
                raise ToolExecutionError(f"Weather error: No location found for '{city}'. Please ask the user to clarify the city name.")
            loc = results[0]
            data = await _http_get_json(
                agent,
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )
            return json.dumps({"location": f"{loc.get('name')}, {loc.get('country')}", "timezone": data.get("timezone"), "current": data.get("current"), "today": data.get("daily")}, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Weather API error: {exc}. Let the user know the weather service failed.")

    @r.register(
        "translate_text",
        "Translate text into a target language.",
        {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]},
    )
    async def _translate(text: str, target_language: str):
        try:
            return await agent.translate_text(text, target_language)
        except Exception as exc:
            raise ToolExecutionError(f"Translation error: {exc}")

    @r.register(
        "fetch_url",
        "Fetch a public HTTP/HTTPS URL and extract readable text.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    )
    async def _fetch_url(url: str):
        try:
            status_code, headers, body, final_url = await agent._pinned_http_get(url)
            if status_code >= 400:
                raise ToolExecutionError(f"URL fetch error: the website returned HTTP {status_code}. Inform the user that the website couldn't be reached.")
            content_type = headers.get("content-type", "")
            text = agent._decode_http_body(body, content_type, 15000)
            if "text/html" in content_type:
                text = _clean_text(text)
            return json.dumps({"url": final_url, "content_type": content_type, "text": text}, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"URL fetch error: {exc}. Inform the user that the website couldn't be reached.")

    # -----------------------------------------------------------------------
    # Gmail
    # -----------------------------------------------------------------------
    @r.register(
        "gmail_list",
        "List recent Gmail messages. Use this first to obtain real message_ids.",
        {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": []},
    )
    async def _gmail_list(query: str = "", max_results: int = 3):
        try:
            data = await asyncio.to_thread(agent.google.gmail_list, query, max_results)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Gmail error: {exc}")

    @r.register(
        "gmail_read",
        "Read a Gmail message by message_id (obtain it from gmail_list first).",
        {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]},
    )
    async def _gmail_read(message_id: str):
        try:
            data = await asyncio.to_thread(agent.google.gmail_read, message_id)
            return json.dumps(data, ensure_ascii=False)[:20000]
        except Exception as exc:
            raise ToolExecutionError(f"Gmail read error: Invalid message ID or API failure. {exc}")

    @r.register(
        "gmail_send_attachment",
        "Download a Gmail attachment and deliver it in the chat (IDs must come from gmail_read).",
        {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},
                "attachment_id": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["message_id", "attachment_id", "filename"],
        },
        side_effect=True,
    )
    async def _gmail_send_attachment(message_id: str, attachment_id: str, filename: str):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            file_bytes = await asyncio.to_thread(
                agent.google.gmail_download_attachment, message_id, attachment_id
            )
            await agent.bot.send_document(
                chat_id=agent.chat_id,
                document=file_bytes,
                filename=filename,
                caption=f"🌸 Here is the attachment from your email: <b>{filename}</b>",
                parse_mode="HTML",
            )
            return f"Success! The file '{filename}' was sent to the user in the chat."
        except Exception as exc:
            raise ToolExecutionError(f"Failed to download or send attachment: {exc}")

    @r.register(
        "gmail_send",
        "Send an email. New recipients require explicit confirmation.",
        {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Plain-text body."},
                "confirm": {"type": "boolean", "description": "Set true only after the user confirms a new recipient."},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _gmail_send(
        to: str,
        subject: str,
        body: str,
        confirm: bool = False,
    ):
        state = await agent.self_state.get(agent.telegram_id)
        prefs = dict(state.get("known_user_prefs") or {})
        recent_recipients = [
            str(x).lower()
            for x in (prefs.get("gmail_recent_recipients") or [])
        ]
        normalized_to = str(to).strip().lower()
        recipient_is_known = normalized_to in recent_recipients

        if not recipient_is_known and not confirm:
            await agent.continuity.set(
                agent.telegram_id,
                topic="gmail",
                intent="gmail_send_confirmation_required",
                pending_confirmation={
                    "action": "gmail_send",
                    "to": to,
                    "subject": subject,
                    "body": body,
                },
            )
            return json.dumps({
                "status": "confirmation_required",
                "to": to,
                "reason": "The recipient differs from recent recipients.",
            }, ensure_ascii=False)

        try:
            data = await asyncio.to_thread(agent.google.gmail_send, to, subject, body)
            updated = [normalized_to] + [x for x in recent_recipients if x != normalized_to]
            prefs["gmail_recent_recipients"] = updated[:10]
            await agent.self_state.update(
                agent.telegram_id,
                known_user_prefs=prefs,
            )
            if confirm:
                await agent.continuity.clear_confirmation(agent.telegram_id)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Gmail send error: {exc}")

    # -----------------------------------------------------------------------
    # Calendar
    # -----------------------------------------------------------------------
    @r.register(
        "calendar_list",
        "List upcoming Google Calendar events.",
        {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 30}, "max_results": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": []},
    )
    async def _calendar_list(days: int = 7, max_results: int = 20):
        try:
            data = await asyncio.to_thread(agent.google.calendar_list, days, max_results)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Calendar error: {exc}")

    @r.register(
        "calendar_create",
        "Create a Google Calendar event (ISO-8601 timestamps). Not for personal reminders.",
        {"type": "object", "properties": {"summary": {"type": "string"}, "start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "description": {"type": "string"}}, "required": ["summary", "start_iso", "end_iso"]},
        side_effect=True,
    )
    async def _calendar_create(summary: str, start_iso: str, end_iso: str, description: str = ""):
        try:
            data = await asyncio.to_thread(agent.google.calendar_create, summary, start_iso, end_iso, description)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Calendar create error: {exc}")

    # -----------------------------------------------------------------------
    # GitHub
    # -----------------------------------------------------------------------
    @r.register(
        "github_list_repos",
        "List repositories for the connected GitHub account.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": []},
    )
    async def _github_repos(limit: int = 20):
        try:
            data = await asyncio.to_thread(agent.github.list_repos, limit)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"GitHub error: {exc}")

    @r.register(
        "github_list_issues",
        "List GitHub issues for a repository.",
        {"type": "object", "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "state": {"type": "string", "enum": ["open", "closed", "all"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["owner", "repo"]},
    )
    async def _github_issues(owner: str, repo: str, state: str = "open", limit: int = 20):
        try:
            data = await asyncio.to_thread(agent.github.list_issues, owner, repo, state, limit)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"GitHub issues error: {exc}")

    @r.register(
        "github_create_issue",
        "Create a GitHub issue.",
        {"type": "object", "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["owner", "repo", "title"]},
        side_effect=True,
    )
    async def _github_create_issue(owner: str, repo: str, title: str, body: str = ""):
        try:
            data = await asyncio.to_thread(agent.github.create_issue, owner, repo, title, body)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"GitHub create issue error: {exc}")

    @r.register(
        "connect_google",
        "Start Google OAuth setup. Use only when the user explicitly asks to connect Google.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        side_effect=True,
    )
    async def _connect_google():
        try:
            return await asyncio.to_thread(agent.google.authorize)
        except Exception as exc:
            raise ToolExecutionError(f"Google authorization error: {exc}")

    # -----------------------------------------------------------------------
    # User profile
    # -----------------------------------------------------------------------
    @r.register(
        "update_user_profile",
        "Save identity facts. Pass only the fields that changed; omitted fields are left untouched.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "date_of_birth": {"type": "string", "description": "ISO YYYY-MM-DD."},
                "education": {"type": "string"},
                "interests": {"type": "string"},
                "location": {"type": "string"},
                "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata."},
                "preferences": {"type": "string"},
            },
            "required": [],
        },
        side_effect=True,
    )
    async def _update_user_profile(
        name: str = None,
        date_of_birth: str = None,
        education: str = None,
        interests: str = None,
        location: str = None,
        timezone: str = None,
        preferences: str = None,
    ):
        updates = {
            "name": name,
            "date_of_birth": date_of_birth,
            "education": education,
            "interests": interests,
            "location": location,
            "timezone": timezone,
            "preferences": preferences,
        }
        updates = {k: v for k, v in updates.items() if v}

        if not updates:
            return "No profile fields were provided to update."

        if "date_of_birth" in updates:
            from dateutil.parser import isoparse
            try:
                isoparse(updates["date_of_birth"])
            except Exception:
                raise ToolExecutionError(
                    f"Could not understand date_of_birth '{updates['date_of_birth']}'. "
                    "Please provide it as YYYY-MM-DD."
                )

        await agent.user_repo.update_profile(agent.telegram_id, updates)
        return f"Profile updated: {', '.join(updates.keys())}. Nothing else was changed."

    # -----------------------------------------------------------------------
    # Google Docs / Drive
    # -----------------------------------------------------------------------
    @r.register(
        "docs_read",
        "Read a Google Doc by document_id.",
        {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
    )
    async def _docs_read(document_id: str):
        try:
            data = await asyncio.to_thread(agent.google.docs_read, document_id)
            return data[:15000]
        except Exception as exc:
            raise ToolExecutionError(f"Google Docs error: {exc}")

    @r.register(
        "drive_list",
        "Search or list Google Drive files.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Optional search query."}, "max_results": {"type": "integer"}}, "required": []},
    )
    async def _drive_list(query: str = "", max_results: int = 10):
        try:
            data = await asyncio.to_thread(agent.google.drive_list, query, max_results)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Google Drive error: {exc}")

    # -----------------------------------------------------------------------
    # Maps (free OpenStreetMap / Nominatim / OSRM)
    # -----------------------------------------------------------------------
    @r.register(
        "get_map_image",
        "Get an OpenStreetMap view (lat/lon + map URL) for a location.",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "minLength": 2, "maxLength": 250},
                "zoom": {"type": "integer", "minimum": 1, "maximum": 19},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    )
    async def _get_map_image(location: str, zoom: int = 14):
        try:
            geo = await _http_get_json(
                agent,
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location,
                    "format": "jsonv2",
                    "limit": 1,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "SakuraAI/1.0"},
            )
            if not geo:
                return json.dumps({"ok": False, "error": f"Location not found: {location}"}, ensure_ascii=False)
            lat = geo[0]["lat"]
            lon = geo[0]["lon"]
            return json.dumps(
                {
                    "location": location,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "map_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}",
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Map lookup error: {exc}")

    @r.register(
        "get_directions",
        "Driving route (distance, duration, key steps) between two places.",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "minLength": 2, "maxLength": 250},
                "destination": {"type": "string", "minLength": 2, "maxLength": 250},
            },
            "required": ["origin", "destination"],
            "additionalProperties": False,
        },
        timeout=30,
    )
    async def _get_directions(origin: str, destination: str):
        try:
            async def geocode(place: str):
                data = await _http_get_json(
                    agent,
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": place, "format": "jsonv2", "limit": 1},
                    headers={"User-Agent": "SakuraAI/1.0"},
                )
                if not data:
                    raise ValueError(f"Location not found: {place}")
                return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", place)

            o_lat, o_lon, o_name = await geocode(origin)
            d_lat, d_lon, d_name = await geocode(destination)

            route = await _http_get_json(
                agent,
                f"https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}",
                params={"overview": "false", "steps": "true"},
            )
            routes = route.get("routes", [])
            if not routes:
                raise ToolExecutionError("Directions error: no driving route found.")

            best = routes[0]
            steps = []
            for leg in best.get("legs", []):
                for step in leg.get("steps", [])[:8]:
                    maneuver = step.get("maneuver", {})
                    instruction = maneuver.get("type", "continue").replace("_", " ")
                    name = step.get("name")
                    text = instruction if not name else f"{instruction} onto {name}"
                    steps.append(text)

            return json.dumps(
                {
                    "origin": o_name,
                    "destination": d_name,
                    "distance_km": round(best.get("distance", 0) / 1000, 2),
                    "duration_minutes": round(best.get("duration", 0) / 60, 1),
                    "key_steps": steps,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            raise ToolExecutionError(f"Directions error: {exc}")

    @r.register(
        "search_places",
        "Search for places via OpenStreetMap Nominatim.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 250},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        timeout=25,
    )
    async def _search_places(query: str, max_results: int = 5):
        try:
            data = await _http_get_json(
                agent,
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": max(1, min(max_results, 8)),
                    "addressdetails": 1,
                },
                headers={"User-Agent": "SakuraAI/1.0"},
            )
            results = [
                {
                    "name": item.get("name") or item.get("display_name", "").split(",")[0],
                    "address": item.get("display_name"),
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                    "osm_type": item.get("osm_type"),
                    "osm_id": item.get("osm_id"),
                }
                for item in data
            ]
            return json.dumps(results, ensure_ascii=False)
        except Exception as exc:
            raise ToolExecutionError(f"Places search error: {exc}")

    @r.register(
        "get_map_link",
        "Generate an OpenStreetMap search link for a place.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 250},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    async def _get_map_link(query: str):
        safe_q = quote(query)
        return json.dumps(
            {
                "query": query,
                "url": f"https://www.openstreetmap.org/search?query={safe_q}",
            },
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # Notes / memory
    # -----------------------------------------------------------------------
    @r.register(
        "save_note",
        "Save a note by title. Reusing the same title updates the existing note.",
        {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]},
        side_effect=True,
    )
    async def _save_note(title: str, content: str):
        note_id, action = await agent.notes.upsert_by_title(agent.telegram_id, title, content)
        return f"Note '{title}' {action} (id: {note_id})."

    @r.register(
        "search_notes",
        "Search notes/memories (hybrid keyword + semantic).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    async def _search_notes(query: str, limit: int = 10):
        docs = await agent.notes.hybrid_search(
            agent.telegram_id,
            query,
            limit=max(1, min(limit, 20)),
        )
        if not docs:
            return f"No matching notes or memories found for '{query}'."
        return json.dumps(
            [
                {
                    "id": str(d["_id"]),
                    "title": d["title"],
                    "content": d["content"],
                }
                for d in docs
            ],
            ensure_ascii=False,
        )

    @r.register(
        "recent_notes",
        "List recently saved notes.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": []},
    )
    async def _recent_notes(limit: int = 10):
        docs = await agent.notes.recent(agent.telegram_id, max(1, min(limit, 20)))
        if not docs:
            return "No notes saved yet."
        return json.dumps([{"id": str(d["_id"]), "title": d["title"], "content": d["content"]} for d in docs], ensure_ascii=False)

    @r.register(
        "update_note",
        "Update a note (note_id must come from search_notes/recent_notes).",
        {
            "type": "object",
            "properties": {
                "note_id": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["note_id", "title", "content"],
        },
        side_effect=True,
    )
    async def _update_note(note_id: str, title: str, content: str):
        success = await agent.notes.update(note_id, agent.telegram_id, title, content)
        if not success:
            raise ToolExecutionError("Failed to update: Invalid ID or note not found.")
        return "Successfully updated the memory."

    @r.register(
        "delete_note",
        "Delete a note. Notes older than 30 days require confirm=true.",
        {
            "type": "object",
            "properties": {
                "note_id": {"type": "string"},
                "confirm": {"type": "boolean"},
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _delete_note(note_id: str, confirm: bool = False):
        doc = await agent.notes.get(note_id, agent.telegram_id)
        if not doc:
            raise ToolExecutionError("Failed to delete: Invalid ID or note not found.")

        created_at = doc.get("created_at")
        is_old = False
        if hasattr(created_at, "tzinfo"):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            is_old = (datetime.now(timezone.utc) - created_at.astimezone(timezone.utc)).days >= 30

        if is_old and not confirm:
            await agent.continuity.set(
                agent.telegram_id,
                topic="memory",
                intent="delete_note_confirmation_required",
                active_entity_type="note",
                active_entity_id=note_id,
                pending_confirmation={
                    "action": "delete_note",
                    "note_id": note_id,
                    "title": doc.get("title", ""),
                },
            )
            return json.dumps({
                "status": "confirmation_required",
                "note_id": note_id,
                "title": doc.get("title", ""),
                "reason": "This note is older than 30 days.",
            }, ensure_ascii=False)

        success = await agent.notes.delete(note_id, agent.telegram_id)
        if success:
            await agent.continuity.clear_confirmation(agent.telegram_id)
        if not success:
            raise ToolExecutionError("Failed to delete: Invalid ID or note not found.")
        return "Successfully deleted the memory."

    # -----------------------------------------------------------------------
    # Reminders
    # -----------------------------------------------------------------------
    @r.register(
        "set_reminder",
        "Create a fixed-time Telegram reminder (ISO-8601 run_at). For condition-based watches use create_watch.",
        {"type": "object", "properties": {"text": {"type": "string"}, "run_at": {"type": "string", "description": "ISO-8601 datetime"}}, "required": ["text", "run_at"]},
        side_effect=True,
    )
    async def _set_reminder(text: str, run_at: str):
        from dateutil.parser import isoparse
        try:
            when = isoparse(run_at)
            if when.tzinfo is None:
                when = when.replace(tzinfo=agent.tz)
            when = when.astimezone(agent.tz)
            if when <= agent.now():
                raise ToolExecutionError("Reminder error: The time provided is in the past. Please provide a future time.")

            reminder_id = await agent.reminders.create(agent.telegram_id, agent.chat_id, text, when.astimezone(timezone.utc))
            if hasattr(agent, "schedule_reminder"):
                scheduled = agent.schedule_reminder(reminder_id, agent.chat_id, text, when)
                if hasattr(scheduled, "__await__"):
                    await scheduled
            return f"Success! Telegram reminder created for {when.strftime('%d %b %Y at %I:%M %p %Z')}."
        except Exception as exc:
            raise ToolExecutionError(f"Reminder error: Could not parse time. Ensure ISO-8601 format. Details: {exc}")

    @r.register(
        "list_reminders",
        "List pending reminders.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _list_reminders():
        docs = await agent.reminders.get_user_pending(agent.telegram_id)
        if not docs:
            return "No pending reminders."

        reminders_list = []
        for d in docs:
            local_time = d["run_at"].replace(tzinfo=timezone.utc).astimezone(agent.tz)
            reminders_list.append({
                "id": str(d["_id"]),
                "text": d["text"],
                "run_at": local_time.strftime("%A, %d %B %Y at %I:%M %p %Z"),
            })
        return json.dumps(reminders_list, ensure_ascii=False)

    @r.register(
        "delete_reminder",
        "Cancel a pending reminder (reminder_id from list_reminders).",
        {"type": "object", "properties": {"reminder_id": {"type": "string"}}, "required": ["reminder_id"]},
        side_effect=True,
    )
    async def _delete_reminder(reminder_id: str):
        success = await agent.reminders.delete(reminder_id, agent.telegram_id)

        if success:
            if hasattr(agent, "cancel_reminder"):
                try:
                    result = agent.cancel_reminder(reminder_id)
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass
            return "Successfully canceled the reminder."

        raise ToolExecutionError("Failed to cancel: Invalid ID or reminder already sent.")

    @r.register(
        "edit_reminder",
        "Edit a pending reminder's text and/or time (reminder_id from list_reminders).",
        {
            "type": "object",
            "properties": {
                "reminder_id": {"type": "string"},
                "text": {"type": "string", "minLength": 1, "maxLength": 1000},
                "run_at": {"type": "string", "description": "Future ISO-8601 datetime."},
            },
            "required": ["reminder_id"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _edit_reminder(reminder_id: str, text: str = None, run_at: str = None):
        if text is None and run_at is None:
            return "Please provide new reminder text, a new time, or both."

        from dateutil.parser import isoparse

        when = None
        if run_at is not None:
            try:
                when = isoparse(run_at)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=agent.tz)
                when = when.astimezone(agent.tz)
            except Exception as exc:
                raise ToolExecutionError(f"Reminder edit error: Could not parse time: {exc}")

            if when <= agent.now():
                raise ToolExecutionError("Reminder edit error: The new time must be in the future.")

        try:
            success = await agent.reminders.update(
                reminder_id,
                agent.telegram_id,
                text=text,
                run_at=when.astimezone(timezone.utc) if when is not None else None,
            )
            if not success:
                raise ToolExecutionError(
                    "Failed to update: invalid ID, reminder not found, "
                    "or reminder is no longer pending."
                )

            if hasattr(agent, "cancel_reminder") and hasattr(agent, "schedule_reminder"):
                try:
                    docs = await agent.reminders.get_user_pending(agent.telegram_id)
                    updated = next(
                        (d for d in docs if str(d.get("_id")) == reminder_id),
                        None,
                    )
                    if updated is not None:
                        stored_run_at = updated.get("run_at")
                        if stored_run_at is not None:
                            if stored_run_at.tzinfo is None:
                                stored_run_at = stored_run_at.replace(tzinfo=timezone.utc)
                            local_run_at = stored_run_at.astimezone(agent.tz)

                            cancel_result = agent.cancel_reminder(reminder_id)
                            if hasattr(cancel_result, "__await__"):
                                await cancel_result

                            schedule_result = agent.schedule_reminder(
                                reminder_id,
                                agent.chat_id,
                                updated.get("text", ""),
                                local_run_at,
                            )
                            if hasattr(schedule_result, "__await__"):
                                await schedule_result
                except Exception:
                    pass

            if when is not None:
                return (
                    "Successfully updated the reminder. New time: "
                    f"{when.strftime('%d %b %Y at %I:%M %p %Z')}."
                )
            return "Successfully updated the reminder."
        except Exception as exc:
            raise ToolExecutionError(f"Reminder edit error: {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # Persistent watches
    # -----------------------------------------------------------------------
    @r.register(
        "create_watch",
        "Persistent background monitor (anime / product / website). Not a fixed-time reminder — use set_reminder for that.",
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["anime", "product", "website"]},
                "target": {"type": "string", "minLength": 2, "maxLength": 200},
                "condition": {"type": "string", "minLength": 2, "maxLength": 500},
                "query": {"type": "string", "maxLength": 500},
                "url": {"type": "string", "maxLength": 1000},
                "interval_minutes": {"type": "integer", "minimum": 15, "maximum": 1440, "description": "Defaults: anime/website 360, product 60."},
            },
            "required": ["kind", "target", "condition"],
            "additionalProperties": False,
        },
        side_effect=True,
        timeout=20,
    )
    async def _create_watch(
        kind: str,
        target: str,
        condition: str,
        interval_minutes: int | None = None,
        query: str = "",
        url: str = "",
    ):
        kind = (kind or "").strip().lower()
        target = (target or "").strip()
        condition = (condition or "").strip()
        query = (query or "").strip()
        url = (url or "").strip()

        if kind == "website" and not url:
            raise ToolExecutionError("Watch error: a website watch requires a URL.")
        if kind == "product" and not url and not query:
            query = f"{target} {condition}"
        if kind == "anime" and not query:
            query = f"{target} {condition} official announcement"

        if interval_minutes is None:
            interval_minutes = 360 if kind in {"anime", "website"} else 60

        current = await agent.watches.count_user(agent.telegram_id)
        if current >= 20:
            return "Watch limit reached: you can have up to 20 active watches."

        existing = await agent.watches.list_user(agent.telegram_id)
        canonical = (kind, target.casefold(), condition.casefold(), query.casefold(), url.casefold())
        for item in existing:
            if not item.get("enabled", True):
                continue
            existing_key = (
                str(item.get("kind") or "").lower(),
                str(item.get("target") or "").casefold(),
                str(item.get("condition") or "").casefold(),
                str(item.get("query") or "").casefold(),
                str(item.get("url") or "").casefold(),
            )
            if existing_key == canonical:
                return json.dumps({
                    "status": "already_active",
                    "watch_id": str(item.get("_id")),
                    "kind": item.get("kind"),
                    "target": item.get("target"),
                    "condition": item.get("condition"),
                    "interval_minutes": item.get("interval_minutes"),
                }, ensure_ascii=False)

        watch_id = await agent.watches.create(
            agent.telegram_id,
            agent.chat_id,
            kind=kind,
            target=target,
            condition=condition,
            query=query,
            url=url,
            interval_minutes=max(15, min(int(interval_minutes), 1440)),
        )
        agent.ensure_watcher_worker()
        await agent.continuity.set(
            agent.telegram_id,
            topic="watch",
            intent="create_watch",
            active_entity_type="watch",
            active_entity_id=watch_id,
        )
        return json.dumps(
            {
                "watch_id": watch_id,
                "kind": kind,
                "target": target,
                "condition": condition,
                "interval_minutes": max(15, min(int(interval_minutes), 1440)),
                "status": "active",
            },
            ensure_ascii=False,
        )

    @r.register(
        "list_watches",
        "List the user's persistent background watches.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _list_watches():
        docs = await agent.watches.list_user(agent.telegram_id)
        if not docs:
            return "No active or saved watches."
        rows = []
        for d in docs:
            next_check = d.get("next_check_at")
            if hasattr(next_check, "astimezone"):
                next_check_text = next_check.astimezone(agent.tz).strftime("%d %b %Y at %I:%M %p %Z")
            else:
                next_check_text = "unknown"
            rows.append(
                {
                    "id": str(d["_id"]),
                    "kind": d.get("kind"),
                    "target": d.get("target"),
                    "condition": d.get("condition"),
                    "interval_minutes": d.get("interval_minutes"),
                    "enabled": bool(d.get("enabled", True)),
                    "next_check": next_check_text,
                    "last_checked": (
                        d.get("last_checked_at").astimezone(agent.tz).strftime("%d %b %Y at %I:%M %p %Z")
                        if hasattr(d.get("last_checked_at"), "astimezone") else None
                    ),
                    "last_error": d.get("last_error"),
                    "failure_count": int(d.get("failure_count") or 0),
                    "notification_count": int(d.get("notification_count") or 0),
                    "last_notified_at": (
                        d.get("last_notified_at").astimezone(agent.tz).strftime("%d %b %Y at %I:%M %p %Z")
                        if hasattr(d.get("last_notified_at"), "astimezone") else None
                    ),
                    "acknowledged_at": (
                        d.get("acknowledged_at").astimezone(agent.tz).strftime("%d %b %Y at %I:%M %p %Z")
                        if hasattr(d.get("acknowledged_at"), "astimezone") else None
                    ),
                }
            )
        if len(rows) == 1:
            await agent.continuity.set(
                agent.telegram_id,
                topic="watch",
                intent="list_watches",
                active_entity_type="watch",
                active_entity_id=rows[0]["id"],
            )
        return json.dumps(rows, ensure_ascii=False)

    @r.register(
        "update_watch",
        "Update a watch's interval or condition/query/url (watch_id from list_watches).",
        {
            "type": "object",
            "properties": {
                "watch_id": {"type": "string"},
                "interval_minutes": {"type": "integer", "minimum": 15, "maximum": 1440},
                "condition": {"type": "string", "minLength": 2, "maxLength": 500},
                "query": {"type": "string", "maxLength": 500},
                "url": {"type": "string", "maxLength": 1000},
            },
            "required": ["watch_id"],
            "additionalProperties": False,
        },
        side_effect=True,
        timeout=20,
    )
    async def _update_watch(
        watch_id: str,
        interval_minutes: int | None = None,
        condition: str | None = None,
        query: str | None = None,
        url: str | None = None,
    ):
        if interval_minutes is None and condition is None and query is None and url is None:
            raise ToolExecutionError("Watch update error: provide at least one setting to change.")
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ToolExecutionError("Watch update error: URL must be a valid HTTP/HTTPS URL.")
        normalized_interval = None if interval_minutes is None else max(15, min(int(interval_minutes), 1440))
        updated = await agent.watches.update_config(
            watch_id,
            agent.telegram_id,
            interval_minutes=normalized_interval,
            condition=condition,
            query=query,
            url=url,
        )
        if not updated:
            raise ToolExecutionError("Watch update failed: watch not found, invalid watch ID, or no settings changed.")
        next_check = updated.get("next_check_at")
        next_check_text = (
            next_check.astimezone(agent.tz).strftime("%d %b %Y at %I:%M %p %Z")
            if hasattr(next_check, "astimezone") else "unknown"
        )
        await agent.continuity.set(
            agent.telegram_id,
            topic="watch",
            intent="update_watch",
            active_entity_type="watch",
            active_entity_id=str(updated.get("_id")),
        )
        return json.dumps({
            "status": "updated",
            "watch_id": str(updated.get("_id")),
            "kind": updated.get("kind"),
            "target": updated.get("target"),
            "condition": updated.get("condition"),
            "interval_minutes": updated.get("interval_minutes"),
            "next_check": next_check_text,
        }, ensure_ascii=False)

    @r.register(
        "delete_watch",
        "Delete a watch (watch_id from list_watches).",
        {"type": "object", "properties": {"watch_id": {"type": "string"}}, "required": ["watch_id"], "additionalProperties": False},
        side_effect=True,
    )
    async def _delete_watch(watch_id: str):
        success = await agent.watches.delete(watch_id, agent.telegram_id)
        if success:
            await agent.continuity.set(
                agent.telegram_id,
                topic="watch",
                intent="delete_watch",
                clear_active_entity=True,
            )
        if not success:
            raise ToolExecutionError("Failed to stop: invalid watch ID or watch not found.")
        return "Successfully stopped the watch."

    @r.register(
        "acknowledge_watch_hit",
        "Mark the latest alert from a watch as handled.",
        {
            "type": "object",
            "properties": {"watch_id": {"type": "string"}},
            "required": ["watch_id"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _acknowledge_watch_hit(watch_id: str):
        success = await agent.watches.acknowledge_hit(watch_id, agent.telegram_id)
        if not success:
            raise ToolExecutionError("Could not acknowledge that watch notification.")
        return "Watch notification acknowledged."

    # -----------------------------------------------------------------------
    # Telegram media
    # -----------------------------------------------------------------------
    @r.register(
        "send_telegram_media",
        "Send a stored photo/video/document/voice by file_id.",
        {
            "type": "object",
            "properties": {
                "file_type": {"type": "string", "enum": ["photo", "video", "document", "voice"]},
                "file_id": {"type": "string"},
            },
            "required": ["file_type", "file_id"],
        },
        side_effect=True,
    )
    async def _send_telegram_media(file_type: str, file_id: str):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            if file_type == "photo":
                await agent.bot.send_photo(chat_id=agent.chat_id, photo=file_id)
            elif file_type == "video":
                await agent.bot.send_video(chat_id=agent.chat_id, video=file_id)
            elif file_type == "document":
                await agent.bot.send_document(chat_id=agent.chat_id, document=file_id)
            elif file_type == "voice":
                await agent.bot.send_voice(chat_id=agent.chat_id, voice=file_id)
            return f"Success! The {file_type} was sent to the chat."
        except Exception as exc:
            raise ToolExecutionError(f"Failed to send {file_type}: {exc}")

    @r.register(
        "analyze_image",
        "Analyze the image on the current or replied-to message.",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 2, "maxLength": 1000},
            },
            "required": [],
            "additionalProperties": False,
        },
        timeout=90,
    )
    async def _analyze_image(prompt: str = "Describe the image and read any useful text from it."):
        if agent.gemini is None or genai_types is None:
            raise ToolExecutionError("Vision is unavailable because GEMINI_API_KEY is not configured.")
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Vision error: Telegram Bot instance is not connected.")

        try:
            message = _resolve_media_message(agent)
            meta = _media_descriptor(message)
            if meta["kind"] != "photo":
                mime = meta["mime_type"]
                if not mime.startswith("image/"):
                    raise ToolExecutionError("Vision error: the current/replied file is not an image.")

            tg_file = await agent.bot.get_file(meta["file_id"])
            image_bytes = bytes(await tg_file.download_as_bytearray())
            if len(image_bytes) > 15 * 1024 * 1024:
                raise ToolExecutionError("Vision error: image is larger than 15 MB.")

            response = await agent.gemini.aio.models.generate_content(
                model=getattr(agent, "MODEL_GEMINI", agent.GEMINI_MODELS[0]),
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_text(text=prompt),
                            genai_types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=meta["mime_type"],
                            ),
                        ],
                    )
                ],
            )
            return response.text or "The vision model returned no text."
        except Exception as exc:
            raise ToolExecutionError(f"Vision processing error: {type(exc).__name__}: {exc}")

    @r.register(
        "analyze_document",
        "Read/analyze the document on the current or replied-to message.",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 2, "maxLength": 1500},
            },
            "required": [],
            "additionalProperties": False,
        },
        timeout=120,
    )
    async def _analyze_document(prompt: str = "Summarize the document and extract the most important information."):
        if agent.gemini is None or genai_types is None:
            raise ToolExecutionError("Document analysis is unavailable because GEMINI_API_KEY is not configured.")
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Document analysis error: Telegram Bot instance is not connected.")

        try:
            message = _resolve_media_message(agent)
            meta = _media_descriptor(message)
            if meta["kind"] != "document":
                raise ToolExecutionError("Document analysis error: the current/replied message is not a document.")

            tg_file = await agent.bot.get_file(meta["file_id"])
            doc_bytes = bytes(await tg_file.download_as_bytearray())
            if len(doc_bytes) > 15 * 1024 * 1024:
                raise ToolExecutionError("Document analysis error: document is larger than 15 MB.")

            response = await agent.gemini.aio.models.generate_content(
                model=getattr(agent, "MODEL_GEMINI", agent.GEMINI_MODELS[0]),
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_text(text=prompt),
                            genai_types.Part.from_bytes(
                                data=doc_bytes,
                                mime_type=meta["mime_type"],
                            ),
                        ],
                    )
                ],
            )
            return response.text or "The document model returned no text."
        except Exception as exc:
            raise ToolExecutionError(f"Document processing error: {type(exc).__name__}: {exc}")

    @r.register(
        "inspect_telegram_context",
        "Get metadata for the current Telegram message (sender, chat, forward origin).",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _inspect_telegram_context():
        msg = getattr(agent, "current_message", None)
        if not msg:
            return json.dumps({
                "telegram_id": agent.telegram_id,
                "chat_id": agent.chat_id,
                "note": "Raw message object not cached; showing active session IDs.",
            }, ensure_ascii=False)

        user = msg.from_user
        meta = {
            "sender": {
                "id": user.id if user else None,
                "first_name": user.first_name if user else None,
                "last_name": user.last_name if user else None,
                "username": user.username if user else None,
                "is_bot": user.is_bot if user else False,
            },
            "chat": {
                "id": msg.chat.id,
                "type": msg.chat.type,
                "title": getattr(msg.chat, "title", None),
            },
            "message_id": msg.message_id,
            "is_forwarded": bool(
                getattr(msg, "forward_date", None)
                or getattr(msg, "forward_origin", None)
            ),
            "forward_origin": None,
        }

        origin = getattr(msg, "forward_origin", None)
        if origin:
            origin_type = getattr(origin, "type", "unknown")
            origin_info = {"type": origin_type}
            if origin_type == "user" and hasattr(origin, "sender_user"):
                u = origin.sender_user
                origin_info.update({"id": u.id, "name": u.full_name, "username": u.username})
            elif origin_type == "channel" and hasattr(origin, "chat"):
                c = origin.chat
                origin_info.update({"id": c.id, "title": c.title, "username": c.username})
            elif origin_type == "chat" and hasattr(origin, "sender_chat"):
                sc = origin.sender_chat
                origin_info.update({"id": sc.id, "title": sc.title, "username": sc.username})
            elif origin_type == "hidden_user":
                origin_info.update({"sender_user_name": getattr(origin, "sender_user_name", "Hidden Account")})
            meta["forward_origin"] = origin_info

        return json.dumps(meta, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Telegram UI helpers
    # -----------------------------------------------------------------------
    def _build_inline_markup(buttons):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        if buttons is None:
            return None
        if not isinstance(buttons, list):
            raise ValueError("buttons must be a 2D list.")
        keyboard = []
        for row in buttons:
            if not isinstance(row, list):
                raise ValueError("Each inline keyboard row must be a list.")
            kb_row = []
            for button in row:
                if not isinstance(button, dict):
                    raise ValueError("Each button must be an object.")
                label = str(button.get("text") or "").strip()
                url = button.get("url")
                callback_data = button.get("callback_data")
                if not label:
                    raise ValueError("Every inline button needs text.")
                if not url and not callback_data:
                    raise ValueError("Every inline button needs either a URL or callback_data.")
                if url and callback_data:
                    raise ValueError("A button cannot contain both URL and callback_data.")
                if callback_data is not None and len(str(callback_data).encode("utf-8")) > 64:
                    raise ValueError("callback_data must be at most 64 UTF-8 bytes.")
                if url is not None:
                    parsed = urlparse(str(url))
                    if parsed.scheme not in {"http", "https", "tg"}:
                        raise ValueError("Inline button URLs must use http, https, or tg.")
                kb_row.append(
                    InlineKeyboardButton(
                        text=label,
                        url=str(url) if url is not None else None,
                        callback_data=str(callback_data) if callback_data is not None else None,
                    )
                )
            keyboard.append(kb_row)
        return InlineKeyboardMarkup(keyboard)

    def _message_id_from(message):
        return int(getattr(message, "message_id", 0) or 0) if message is not None else 0

    async def _resolve_ui_target():
        current = agent.current_message
        if current is not None:
            mid = _message_id_from(current)
            current_chat = getattr(getattr(current, "chat", None), "id", agent.chat_id)
            markup = getattr(current, "reply_markup", None)
            is_interactive = (
                agent.request_context.interaction_type == "callback"
                or bool(getattr(markup, "inline_keyboard", None))
            )
            if mid and int(current_chat) == int(agent.chat_id) and is_interactive:
                return mid, current, None

        latest = await agent.latest_telegram_ui()
        if latest:
            return int(latest.get("message_id") or 0), None, latest
        return 0, None, None

    @r.register(
        "get_workflow_state",
        "Read stored state for a Telegram workflow by workflow_id.",
        {
            "type": "object",
            "properties": {"workflow_id": {"type": "string", "minLength": 1, "maxLength": 120}},
            "required": ["workflow_id"],
            "additionalProperties": False,
        },
    )
    async def _get_workflow_state(workflow_id: str):
        state = await agent.workflow_state.get(
            agent.telegram_id,
            agent.chat_id,
            workflow_id,
        )
        if not state:
            return json.dumps({"status": "not_found", "workflow_id": workflow_id})
        return json.dumps(state, ensure_ascii=False, default=str)

    @r.register(
        "set_workflow_state",
        "Create or advance a Telegram workflow state.",
        {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1, "maxLength": 120},
                "workflow_type": {"type": "string", "minLength": 1, "maxLength": 80},
                "state": {"type": "string", "minLength": 1, "maxLength": 80},
                "data": {"type": "object"},
            },
            "required": ["workflow_id", "workflow_type", "state"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _set_workflow_state(
        workflow_id: str,
        workflow_type: str,
        state: str,
        data: dict | None = None,
    ):
        await agent.workflow_state.set(
            agent.telegram_id,
            agent.chat_id,
            workflow_id,
            workflow_type,
            state,
            data or {},
        )
        await agent.self_state.update(
            agent.telegram_id,
            active_workflow={
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "state": state,
            },
        )
        return json.dumps({
            "status": "updated",
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "state": state,
            "data": data or {},
            "_suppress_final": False,
        }, ensure_ascii=False)

    @r.register(
        "get_telegram_ui_context",
        "Inspect current/recent Telegram interactive UI state before editing a menu or keyboard.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": [],
            "additionalProperties": False,
        },
        timeout=15,
    )
    async def _get_telegram_ui_context(limit: int = 5):
        try:
            current = agent.current_message
            current_info = None
            current_markup = getattr(current, "reply_markup", None) if current is not None else None
            current_is_interactive = bool(getattr(current_markup, "inline_keyboard", None)) or agent.request_context.interaction_type == "callback"
            if current is not None and _message_id_from(current) and current_is_interactive:
                buttons = []
                for row in getattr(current_markup, "inline_keyboard", []) or []:
                    for button in row:
                        buttons.append({
                            "text": getattr(button, "text", ""),
                            "callback_data": getattr(button, "callback_data", None),
                            "url": getattr(button, "url", None),
                        })
                current_info = {
                    "message_id": _message_id_from(current),
                    "chat_id": getattr(getattr(current, "chat", None), "id", agent.chat_id),
                    "text": getattr(current, "text", None) or getattr(current, "caption", None) or "",
                    "buttons": buttons,
                    "is_callback_message": agent.request_context.interaction_type == "callback",
                }

            recent = await agent.recent_telegram_ui(limit)
            saved = []
            seen = set()
            if current_info:
                seen.add(current_info["message_id"])
            for item in recent:
                mid = int(item.get("message_id") or 0)
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                saved.append({
                    "message_id": mid,
                    "chat_id": int(item.get("chat_id") or agent.chat_id),
                    "text": str(item.get("text") or ""),
                    "buttons": item.get("buttons") or [],
                    "kind": item.get("kind") or "interactive",
                    "updated_at": str(item.get("updated_at") or ""),
                })

            return json.dumps({
                "current": current_info,
                "recent": saved,
                "actions": [
                    "send_inline_keyboard",
                    "edit_telegram_message",
                    "delete_telegram_message",
                    "pin_telegram_message",
                    "unpin_telegram_message",
                ],
            }, ensure_ascii=False, default=str)
        except Exception as exc:
            raise ToolExecutionError(f"Telegram UI context error: {type(exc).__name__}: {exc}")

    @r.register(
        "send_inline_keyboard",
        "Send a new Telegram message with inline buttons.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "url": {"type": "string"},
                                "callback_data": {"type": "string"},
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "kind": {"type": "string", "maxLength": 80},
                "workflow_id": {"type": "string", "maxLength": 120},
                "workflow_type": {"type": "string", "maxLength": 80},
                "workflow_state": {"type": "string", "maxLength": 80},
                "workflow_data": {"type": "object"},
            },
            "required": ["text", "buttons"],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _send_inline_keyboard(
        text: str,
        buttons: list[list[dict]],
        kind: str = "interactive",
        workflow_id: str | None = None,
        workflow_type: str | None = None,
        workflow_state: str = "active",
        workflow_data: dict | None = None,
    ):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            from telegram.constants import ParseMode
            markup = _build_inline_markup(buttons)
            sent = await agent.bot.send_message(
                chat_id=agent.chat_id,
                text=text,
                reply_markup=markup,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
            await agent.remember_telegram_ui(
                sent.message_id,
                text=text,
                buttons=buttons,
                kind=kind,
            )
            if workflow_id:
                await agent.workflow_state.set(
                    agent.telegram_id,
                    agent.chat_id,
                    workflow_id,
                    workflow_type or kind,
                    workflow_state,
                    workflow_data or {},
                )
                await agent.self_state.update(
                    agent.telegram_id,
                    active_workflow={
                        "workflow_id": workflow_id,
                        "workflow_type": workflow_type or kind,
                        "state": workflow_state,
                    },
                )
            return {
                "message_id": sent.message_id,
                "chat_id": agent.chat_id,
                "kind": kind,
                "_suppress_final": True,
                "status": "created",
            }
        except Exception as exc:
            raise ToolExecutionError(f"Failed to send inline keyboard: {type(exc).__name__}: {exc}")

    @r.register(
        "edit_telegram_message",
        "Edit an existing Telegram message: text, buttons, or remove buttons with buttons=[].",
        {
            "type": "object",
            "properties": {
                "message_id": {"type": "integer", "minimum": 1},
                "text": {"type": ["string", "null"]},
                "buttons": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "url": {"type": "string"},
                                "callback_data": {"type": "string"},
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "kind": {"type": "string", "maxLength": 80},
                "workflow_id": {"type": "string", "maxLength": 120},
                "workflow_type": {"type": "string", "maxLength": 80},
                "workflow_state": {"type": "string", "maxLength": 80},
                "workflow_data": {"type": "object"},
            },
            "required": [],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _edit_telegram_message(
        message_id: int | None = None,
        text: str | None = None,
        buttons: list[list[dict]] | None = None,
        kind: str = "interactive",
        workflow_id: str | None = None,
        workflow_type: str | None = None,
        workflow_state: str | None = None,
        workflow_data: dict | None = None,
    ):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        if text is None and buttons is None:
            raise ToolExecutionError("Edit error: provide text, buttons, or buttons=[] to remove the keyboard.")
        try:
            from telegram.constants import ParseMode
            mid, current, saved = await _resolve_ui_target()
            if message_id:
                mid = int(message_id)
                if current is None or _message_id_from(current) != mid:
                    current = None
                    latest_items = await agent.recent_telegram_ui(10)
                    saved = next((x for x in latest_items if int(x.get("message_id") or 0) == mid), None)
            if not mid:
                raise ToolExecutionError("Edit error: no current or remembered Telegram message is available.")

            if buttons is None:
                if current is not None:
                    markup = getattr(current, "reply_markup", None)
                else:
                    stored = (saved or {}).get("buttons") if saved else None
                    markup = _build_inline_markup(stored) if stored is not None else None
            else:
                markup = _build_inline_markup(buttons)

            if text is not None:
                if current is not None:
                    await current.edit_text(
                        text,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=markup,
                        disable_web_page_preview=True,
                    )
                else:
                    await agent.bot.edit_message_text(
                        chat_id=agent.chat_id,
                        message_id=mid,
                        text=text,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=markup,
                        disable_web_page_preview=True,
                    )
            elif buttons is not None:
                if current is not None:
                    await current.edit_reply_markup(reply_markup=markup)
                else:
                    await agent.bot.edit_message_reply_markup(
                        chat_id=agent.chat_id,
                        message_id=mid,
                        reply_markup=markup,
                    )

            effective_text = text
            if effective_text is None:
                if current is not None:
                    effective_text = getattr(current, "text", None) or getattr(current, "caption", None) or ""
                else:
                    effective_text = str((saved or {}).get("text") or "")

            effective_buttons = buttons
            if effective_buttons is None:
                if saved is not None:
                    effective_buttons = saved.get("buttons") or []
                elif current is not None:
                    effective_buttons = []
                    markup_obj = getattr(current, "reply_markup", None)
                    for row in getattr(markup_obj, "inline_keyboard", []) or []:
                        effective_buttons.append([
                            {
                                "text": getattr(b, "text", ""),
                                "callback_data": getattr(b, "callback_data", None),
                                "url": getattr(b, "url", None),
                            }
                            for b in row
                        ])
            await agent.remember_telegram_ui(
                mid,
                text=effective_text or "",
                buttons=effective_buttons or [],
                kind=kind,
            )
            if workflow_id and workflow_state:
                await agent.workflow_state.set(
                    agent.telegram_id,
                    agent.chat_id,
                    workflow_id,
                    workflow_type or kind,
                    workflow_state,
                    workflow_data or {},
                )
                await agent.self_state.update(
                    agent.telegram_id,
                    active_workflow={
                        "workflow_id": workflow_id,
                        "workflow_type": workflow_type or kind,
                        "state": workflow_state,
                    },
                )
            return {
                "message_id": mid,
                "chat_id": agent.chat_id,
                "status": "updated",
                "buttons_changed": buttons is not None,
                "text_changed": text is not None,
                "_suppress_final": True,
            }
        except Exception as exc:
            raise ToolExecutionError(f"Edit error: {type(exc).__name__}: {exc}")

    @r.register(
        "delete_telegram_message",
        "Delete the current or remembered Telegram message.",
        {
            "type": "object",
            "properties": {"message_id": {"type": "integer", "minimum": 1}},
            "required": [],
            "additionalProperties": False,
        },
        side_effect=True,
    )
    async def _delete_telegram_message(message_id: int | None = None):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            mid, current, _saved = await _resolve_ui_target()
            if message_id:
                mid = int(message_id)
                if current is not None and _message_id_from(current) != mid:
                    current = None
            if not mid:
                raise ToolExecutionError("Delete error: no current or remembered Telegram message is available.")
            if current is not None:
                await current.delete()
            else:
                await agent.bot.delete_message(chat_id=agent.chat_id, message_id=mid)
            await agent.forget_telegram_ui(mid)
            return {"message_id": mid, "status": "deleted", "_suppress_final": True}
        except Exception as exc:
            raise ToolExecutionError(f"Delete error: {type(exc).__name__}: {exc}")

    @r.register(
        "pin_telegram_message",
        "Pin the current or remembered message.",
        {"type": "object", "properties": {"message_id": {"type": "integer", "minimum": 1}}, "required": [], "additionalProperties": False},
        side_effect=True,
    )
    async def _pin_telegram_message(message_id: int | None = None):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            mid, current, _saved = await _resolve_ui_target()
            mid = int(message_id) if message_id else mid
            if not mid:
                raise ToolExecutionError("Pin error: no current or remembered message is available.")
            await agent.bot.pin_chat_message(chat_id=agent.chat_id, message_id=mid, disable_notification=True)
            return {"message_id": mid, "status": "pinned", "_suppress_final": True}
        except Exception as exc:
            raise ToolExecutionError(f"Pin error: {type(exc).__name__}: {exc}")

    @r.register(
        "unpin_telegram_message",
        "Unpin the current or remembered message.",
        {"type": "object", "properties": {"message_id": {"type": "integer", "minimum": 1}}, "required": [], "additionalProperties": False},
        side_effect=True,
    )
    async def _unpin_telegram_message(message_id: int | None = None):
        if not hasattr(agent, "bot"):
            raise ToolExecutionError("Error: Telegram Bot instance not connected.")
        try:
            mid, _current, _saved = await _resolve_ui_target()
            mid = int(message_id) if message_id else mid
            if not mid:
                raise ToolExecutionError("Unpin error: no current or remembered message is available.")
            await agent.bot.unpin_chat_message(chat_id=agent.chat_id, message_id=mid)
            return {"message_id": mid, "status": "unpinned", "_suppress_final": True}
        except Exception as exc:
            raise ToolExecutionError(f"Unpin error: {type(exc).__name__}: {exc}")