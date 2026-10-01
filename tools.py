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


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large. Keep it under 1000.")
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero is not allowed.")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_safe_eval(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError("Unsupported mathematical expression.")


def calculate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression or len(expression) > 500:
        raise ValueError("Expression must be 1-500 characters long.")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    return str(result)

# ---------------------------------------------------------------------------
# Network error handling (Using shared agent.http_client for efficiency)
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


async def _http_post_json(agent, url: str, payload: dict, headers: dict | None = None):
    # Do not blindly retry POST requests: some POSTs may be side-effecting.
    try:
        response = await agent.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise Exception("The API request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise Exception(f"API returned an error code: {exc.response.status_code}") from exc


async def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP/HTTPS URLs with a hostname are supported.")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are blocked.")
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("The hostname could not be resolved.") from exc

    for item in infos:
        address = item[4][0]
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("Private or local network targets are blocked.")


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

    @r.register(
        "calculate",
        "Safely evaluate a mathematical expression. Supports basic arithmetic, trig, and logarithms. USE THIS instead of doing math in your head.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    )
    async def _calculate(expression: str):
        try:
            return await asyncio.to_thread(calculate, expression)
        except Exception as exc:
            return f"Calculation error: {exc}. Please adjust your expression and try again."

    @r.register(
        "current_time",
        "Get the exact current date and time in the user's local timezone.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _current_time():
        return agent.now().strftime("%A, %d %B %Y at %I:%M %p %Z")

    @r.register(
        "set_reminder",
        "Create a fixed-time Telegram reminder. For requests like 'watch this anime and tell me when a new season is announced' or 'tell me when this product is back in stock', use create_watch instead of set_reminder.",
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
                return "Reminder error: The time provided is in the past. Please provide a future time."
            
            reminder_id = await agent.reminders.create(agent.telegram_id, agent.chat_id, text, when.astimezone(timezone.utc))
            if hasattr(agent, "schedule_reminder"):
                scheduled = agent.schedule_reminder(reminder_id, agent.chat_id, text, when)
                if hasattr(scheduled, "__await__"):
                    await scheduled
            return f"Success! Telegram reminder created for {when.strftime('%d %b %Y at %I:%M %p %Z')}."
        except Exception as exc:
            return f"Reminder error: Could not parse time. Ensure ISO-8601 format. Details: {exc}"


    @r.register(
        "web_search",
        "Search the public web using DuckDuckGo's free HTML endpoint. Use this for current facts, recent news, or uncertain information.",
        {"type": "object", "properties": {"query": {"type": "string", "minLength": 2, "maxLength": 300}, "max_results": {"type": "integer", "minimum": 1, "maximum": 6}}, "required": ["query"], "additionalProperties": False},
        timeout=25,
    )
    async def _web_search(query: str, max_results: int = 5):
        try:
            response = await agent.http_client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "us-en"},
                headers={"User-Agent": "SakuraAI/1.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            for item in soup.select(".result")[:max(1, min(max_results, 6))]:
                anchor = item.select_one(".result__a")
                snippet = item.select_one(".result__snippet")
                if not anchor:
                    continue
                results.append({
                    "title": anchor.get_text(" ", strip=True),
                    "url": anchor.get("href"),
                    "content": snippet.get_text(" ", strip=True) if snippet else "",
                })
            return json.dumps({"query": query, "results": results}, ensure_ascii=False)
        except Exception as exc:
            return f"Web search error: {exc}"

    @r.register(
        "get_weather",
        "Get current weather and today's forecast for a city using Open-Meteo.",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    async def _weather(city: str):
        try:
            geo = await _http_get_json(agent, "https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1, "language": "en", "format": "json"})
            results = geo.get("results", [])
            if not results:
                return f"Weather error: No location found for '{city}'. Please ask the user to clarify the city name."
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
            return f"Weather API error: {exc}. Let the user know the weather service failed."

    @r.register(
        "translate_text",
        "Translate text into a target language.",
        {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]},
    )
    async def _translate(text: str, target_language: str):
        try:
            return await agent.translate_text(text, target_language)
        except Exception as exc:
            return f"Translation error: {exc}"

    @r.register(
        "fetch_url",
        "Fetch a public HTTP/HTTPS URL and extract readable text. Useful for summarizing links the user sends you.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    )
    async def _fetch_url(url: str):
        try:
            await _assert_public_url(url)
            response = await agent.http_client.get(
                url,
                follow_redirects=False,
                headers={"User-Agent": "SakuraAI/1.0"},
            )
            # Do not automatically follow redirects: validate the redirect target
            # first to prevent localhost/private-network SSRF through a public URL.
            if 300 <= response.status_code < 400:
                location = response.headers.get("location")
                if not location:
                    return "URL error: the website returned a redirect without a target."
                from urllib.parse import urljoin
                target = urljoin(url, location)
                await _assert_public_url(target)
                response = await agent.http_client.get(
                    target,
                    follow_redirects=False,
                    headers={"User-Agent": "SakuraAI/1.0"},
                )

            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                text = _clean_text(response.text)
            else:
                text = response.text[:15000]
            return json.dumps({"url": str(response.url), "content_type": content_type, "text": text}, ensure_ascii=False)
        except Exception as exc:
            return f"URL fetch error: {exc}. Inform the user that the website couldn't be reached."

    @r.register(
        "gmail_list",
        "List recent Gmail messages. ALWAYS USE THIS FIRST to get message_ids before calling gmail_read. Optional query uses standard Gmail search syntax.",
        {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": []},
    )
    async def _gmail_list(query: str = "", max_results: int = 3):
        try:
            data = await asyncio.to_thread(agent.google.gmail_list, query, max_results)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"Gmail error: {exc}"

    @r.register(
        "gmail_read",
        "Read the content of a specific Gmail message. You MUST obtain the message_id from gmail_list first. Never guess or hallucinate IDs.",
        {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]},
    )
    async def _gmail_read(message_id: str):
        try:
            data = await asyncio.to_thread(agent.google.gmail_read, message_id)
            return json.dumps(data, ensure_ascii=False)[:20000]
        except Exception as exc:
            return f"Gmail read error: Invalid message ID or API failure. {exc}"

    @r.register(
        "gmail_send_attachment",
        "Download an email attachment and send it directly to the user in the Telegram chat. You MUST obtain the message_id and attachment_id from gmail_read first.",
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
            return "Error: Telegram Bot instance not connected."
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
            return f"Failed to download or send attachment: {exc}"

    @r.register(
        "gmail_send",
        "Send an email from the connected Gmail account.",
        {
            "type": "object", 
            "properties": {
                "to": {"type": "string"}, 
                "subject": {"type": "string"}, 
                "body": {"type": "string", "description": "The email message body in clean, professional plain text. DO NOT include Telegram HTML tags like <b> or <code>."}
            }, 
            "required": ["to", "subject", "body"]
        },
        side_effect=True,
    )
    async def _gmail_send(to: str, subject: str, body: str):
        try:
            data = await asyncio.to_thread(agent.google.gmail_send, to, subject, body)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"Gmail send error: {exc}"

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
            return f"Calendar error: {exc}"

    @r.register(
        "calendar_create",
        "Create a Google Calendar event. USE THIS FOR: 'Schedule a meeting', 'Block my calendar', or events. DO NOT use this for personal Telegram reminders. Requires ISO-8601 timestamps.",
        {"type": "object", "properties": {"summary": {"type": "string"}, "start_iso": {"type": "string"}, "end_iso": {"type": "string"}, "description": {"type": "string"}}, "required": ["summary", "start_iso", "end_iso"]},
        side_effect=True,
    )
    async def _calendar_create(summary: str, start_iso: str, end_iso: str, description: str = ""):
        try:
            data = await asyncio.to_thread(agent.google.calendar_create, summary, start_iso, end_iso, description)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"Calendar create error: {exc}"

    @r.register(
        "github_list_repos",
        "List repositories accessible to the connected GitHub account.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": []},
    )
    async def _github_repos(limit: int = 20):
        try:
            data = await asyncio.to_thread(agent.github.list_repos, limit)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"GitHub error: {exc}"

    @r.register(
        "github_list_issues",
        "List issues for a GitHub repository.",
        {"type": "object", "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "state": {"type": "string", "enum": ["open", "closed", "all"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["owner", "repo"]},
    )
    async def _github_issues(owner: str, repo: str, state: str = "open", limit: int = 20):
        try:
            data = await asyncio.to_thread(agent.github.list_issues, owner, repo, state, limit)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"GitHub issues error: {exc}"

    @r.register(
        "github_create_issue",
        "Create a GitHub issue in a repository.",
        {"type": "object", "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}}, "required": ["owner", "repo", "title"]},
        side_effect=True,
    )
    async def _github_create_issue(owner: str, repo: str, title: str, body: str = ""):
        try:
            data = await asyncio.to_thread(agent.github.create_issue, owner, repo, title, body)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"GitHub create issue error: {exc}"

    @r.register(
        "connect_google",
        "Start Google OAuth setup for Gmail and Calendar. Use ONLY when the user explicitly asks to connect/authorize Google.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        side_effect=True,
    )
    async def _connect_google():
        try:
            return await asyncio.to_thread(agent.google.authorize)
        except Exception as exc:
            return f"Google authorization error: {exc}"


    @r.register(
        "update_user_profile",
        (
            "Save or update ONE OR MORE specific facts about Senpai's identity. Pass ONLY the "
            "fields that are actually changing — never re-send fields that haven't changed, "
            "each field is stored and updated independently so nothing else gets touched."
        ),
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Senpai's first name / what to call them."},
                "date_of_birth": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD. Convert whatever format the user gave "
                                    "(e.g. '11 nov 2005') into this exact format before saving.",
                },
                "education": {"type": "string", "description": "Current course, college, field of study."},
                "interests": {"type": "string", "description": "Hobbies, technical interests, what they're learning."},
                "location": {"type": "string", "description": "City/state/country they live in."},
                "timezone": {"type": "string", "description": "IANA timezone string, e.g. 'Asia/Kolkata'."},
                "preferences": {"type": "string", "description": "Standing preferences for how you talk/help, e.g. language, tone."},
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

        # Light sanity check on date_of_birth so a malformed date never gets saved silently
        if "date_of_birth" in updates:
            from dateutil.parser import isoparse
            try:
                isoparse(updates["date_of_birth"])
            except Exception:
                return (
                    f"Could not understand date_of_birth '{updates['date_of_birth']}'. "
                    "Please provide it as YYYY-MM-DD."
                )

        await agent.user_repo.update_profile(agent.telegram_id, updates)
        return f"Profile updated: {', '.join(updates.keys())}. Nothing else was changed."


    
    @r.register(
        "docs_read",
        "Read the text content of a Google Doc using its document ID.",
        {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
    )
    async def _docs_read(document_id: str):
        try:
            data = await asyncio.to_thread(agent.google.docs_read, document_id)
            return data[:15000]
        except Exception as exc:
            return f"Google Docs error: {exc}"

    @r.register(
        "drive_list",
        "Search or list files in the user's Google Drive.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Optional search query (e.g., name contains 'ML')"}, "max_results": {"type": "integer"}}, "required": []},
    )
    async def _drive_list(query: str = "", max_results: int = 10):
        try:
            data = await asyncio.to_thread(agent.google.drive_list, query, max_results)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return f"Google Drive error: {exc}"

    # ---------------------------------------------------------------------------
    # Google Maps Suite (Static Maps, Directions, Places, Direct Links)
    # ---------------------------------------------------------------------------

    @r.register(
        "get_map_image",
        "Open a free OpenStreetMap map view for a location. This does not call a paid maps API.",
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
            return f"Map lookup error: {exc}"

    @r.register(
        "get_directions",
        "Get a driving route between two places using free OpenStreetMap geocoding and OSRM routing.",
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
                return "Directions error: no driving route found."

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
            return f"Directions error: {exc}"

    @r.register(
        "search_places",
        "Find nearby or matching places with free OpenStreetMap Nominatim search. Results do not include commercial ratings.",
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
            return f"Places search error: {exc}"

    @r.register(
        "get_map_link",
        "Generate a free OpenStreetMap search link for a place or address.",
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

    # ---------------------------------------------------------------------------
    # CRUD Memory Notes
    # ---------------------------------------------------------------------------

    @r.register(
        "save_note",
        (
            "Save a note under a short, specific title. If a note with the SAME title already "
            "exists, this UPDATES it instead of creating a duplicate — so always reuse the exact "
            "same title when adding to or correcting something you already noted, and use a new, "
            "distinct title for something genuinely new."
        ),
        {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]},
        side_effect=True,
    )
    async def _save_note(title: str, content: str):
        note_id, action = await agent.notes.upsert_by_title(agent.telegram_id, title, content)
        return f"Note '{title}' {action} (id: {note_id})."

    @r.register(
        "search_notes",
        "Search the user's saved notes or long-term memory by keyword.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    async def _search_notes(query: str):
        docs = await agent.notes.search(agent.telegram_id, query)
        if not docs:
            return f"No matching notes or memories found for '{query}'."
        return json.dumps([{"id": str(d["_id"]), "title": d["title"], "content": d["content"]} for d in docs], ensure_ascii=False)

    @r.register(
        "recent_notes",
        "List recently saved notes or memories. Good for getting context on what the user was recently working on.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": []},
    )
    async def _recent_notes(limit: int = 10):
        docs = await agent.notes.recent(agent.telegram_id, max(1, min(limit, 20)))
        if not docs:
            return "No notes saved yet."
        return json.dumps([{"id": str(d["_id"]), "title": d["title"], "content": d["content"]} for d in docs], ensure_ascii=False)

    @r.register(
        "update_note",
        "Update an existing note or memory. You MUST get the note 'id' from search_notes or recent_notes first.",
        {
            "type": "object", 
            "properties": {
                "note_id": {"type": "string"}, 
                "title": {"type": "string"}, 
                "content": {"type": "string"}
            }, 
            "required": ["note_id", "title", "content"]
        },
        side_effect=True,
    )
    async def _update_note(note_id: str, title: str, content: str):
        success = await agent.notes.update(note_id, agent.telegram_id, title, content)
        return f"Successfully updated the memory." if success else "Failed to update: Invalid ID or note not found."

    @r.register(
        "delete_note",
        "Delete a saved note or memory. You MUST get the note 'id' from search_notes or recent_notes first.",
        {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]},
        side_effect=True,
    )
    async def _delete_note(note_id: str):
        success = await agent.notes.delete(note_id, agent.telegram_id)
        return "Successfully deleted the memory." if success else "Failed to delete: Invalid ID or note not found."

    # ---------------------------------------------------------------------------
    # Reminders Management
    # ---------------------------------------------------------------------------
    @r.register(
        "list_reminders",
        "List all of the user's pending scheduled reminders.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _list_reminders():
        docs = await agent.reminders.get_user_pending(agent.telegram_id)
        if not docs:
            return "No pending reminders."
        
        reminders_list = []
        for d in docs:
            # Convert the UTC time back to the user's local timezone for readability
            local_time = d["run_at"].replace(tzinfo=timezone.utc).astimezone(agent.tz)
            reminders_list.append({
                "id": str(d["_id"]),
                "text": d["text"],
                "run_at": local_time.strftime("%A, %d %B %Y at %I:%M %p %Z")
            })
        return json.dumps(reminders_list, ensure_ascii=False)

    @r.register(
        "delete_reminder",
        "Cancel or delete a pending reminder. You MUST get the reminder 'id' from list_reminders first.",
        {"type": "object", "properties": {"reminder_id": {"type": "string"}}, "required": ["reminder_id"]},
        side_effect=True,
    )
    async def _delete_reminder(reminder_id: str):
        # 1. Delete from database
        success = await agent.reminders.delete(reminder_id, agent.telegram_id)
        
        if success:
            # Optional in-process scheduler hook.
            if hasattr(agent, "cancel_reminder"):
                try:
                    result = agent.cancel_reminder(reminder_id)
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass
            return "Successfully canceled the reminder."
            
        return "Failed to cancel: Invalid ID or reminder already sent."


    @r.register(
        "edit_reminder",
        (
            "Edit a pending reminder. You MUST get the reminder 'id' from list_reminders first. "
            "Change its text, scheduled time, or both."
        ),
        {
            "type": "object",
            "properties": {
                "reminder_id": {"type": "string"},
                "text": {"type": "string", "minLength": 1, "maxLength": 1000},
                "run_at": {
                    "type": "string",
                    "description": "Future ISO-8601 datetime in the configured local timezone."
                },
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
                return f"Reminder edit error: Could not parse time: {exc}"

            if when <= agent.now():
                return "Reminder edit error: The new time must be in the future."

        try:
            success = await agent.reminders.update(
                reminder_id,
                agent.telegram_id,
                text=text,
                run_at=when.astimezone(timezone.utc) if when is not None else None,
            )
            if not success:
                return (
                    "Failed to update: invalid ID, reminder not found, "
                    "or reminder is no longer pending."
                )

            # Refresh an optional in-process scheduler if the host application
            # provides scheduler hooks. Fetch the updated reminder first so a
            # text-only edit keeps the original scheduled time.
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
                    # The DB update already succeeded. Scheduler refresh is best-effort.
                    pass

            if when is not None:
                return (
                    "Successfully updated the reminder. New time: "
                    f"{when.strftime('%d %b %Y at %I:%M %p %Z')}."
                )
            return "Successfully updated the reminder."
        except Exception as exc:
            return f"Reminder edit error: {type(exc).__name__}: {exc}"



    # ---------------------------------------------------------------------------
    # Persistent condition watches
    # ---------------------------------------------------------------------------
    @r.register(
        "create_watch",
        (
            "Create a persistent background monitor. Use for requests such as: "
            "watch an anime for an official new-season announcement, monitor a product "
            "until it is back in stock, or watch a specific webpage for a change. "
            "This is NOT a fixed-time reminder. It keeps checking until the condition occurs."
        ),
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["anime", "product", "website"]},
                "target": {"type": "string", "minLength": 2, "maxLength": 200},
                "condition": {"type": "string", "minLength": 2, "maxLength": 500},
                "query": {"type": "string", "maxLength": 500},
                "url": {"type": "string", "maxLength": 1000},
                "interval_minutes": {"type": "integer", "minimum": 15, "maximum": 1440, "description": "How often to check. If omitted: anime 360 min, product 60 min, website 360 min."},
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
            return "Watch error: a website watch requires a URL."
        if kind == "product" and not url and not query:
            query = f"{target} {condition}"
        if kind == "anime" and not query:
            query = f"{target} {condition} official announcement"

        if interval_minutes is None:
            interval_minutes = 360 if kind in {"anime", "website"} else 60

        current = await agent.watches.count_user(agent.telegram_id)
        if current >= 20:
            return "Watch limit reached: you can have up to 20 active watches."

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
        "List the user's persistent background watches and their current status. Use before deleting a watch.",
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
                }
            )
        return json.dumps(rows, ensure_ascii=False)

    @r.register(
        "delete_watch",
        "Delete a persistent background watch. You MUST obtain watch_id from list_watches first.",
        {"type": "object", "properties": {"watch_id": {"type": "string"}}, "required": ["watch_id"], "additionalProperties": False},
        side_effect=True,
    )
    async def _delete_watch(watch_id: str):
        success = await agent.watches.delete(watch_id, agent.telegram_id)
        return "Successfully stopped the watch." if success else "Failed to stop: invalid watch ID or watch not found."

    @r.register(
        "send_telegram_media",
        "Send a saved photo, video, document, or voice note back to the user in the chat using its file_id. You MUST get the file_id from your notes first.",
        {
            "type": "object",
            "properties": {
                "file_type": {"type": "string", "enum": ["photo", "video", "document", "voice"]},
                "file_id": {"type": "string"}
            },
            "required": ["file_type", "file_id"]
        },
        side_effect=True,
    )
    async def _send_telegram_media(file_type: str, file_id: str):
        if not hasattr(agent, "bot"):
            return "Error: Telegram Bot instance not connected."
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
            return f"Failed to send {file_type}: {exc}"


    @r.register(
        "analyze_image",
        "Analyze the image attached to the current Telegram message or the message being replied to. Do not invent file IDs.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 1000,
                    "description": "What to inspect in the image."
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        timeout=90,
    )
    async def _analyze_image(prompt: str = "Describe the image and read any useful text from it."):
        if agent.gemini is None or genai_types is None:
            return "Vision is unavailable because GEMINI_API_KEY is not configured."
        if not hasattr(agent, "bot"):
            return "Vision error: Telegram Bot instance is not connected."

        try:
            message = _resolve_media_message(agent)
            meta = _media_descriptor(message)
            if meta["kind"] != "photo":
                mime = meta["mime_type"]
                if not mime.startswith("image/"):
                    return "Vision error: the current/replied file is not an image."

            tg_file = await agent.bot.get_file(meta["file_id"])
            image_bytes = bytes(await tg_file.download_as_bytearray())
            if len(image_bytes) > 15 * 1024 * 1024:
                return "Vision error: image is larger than 15 MB."

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
            return f"Vision processing error: {type(exc).__name__}: {exc}"

    @r.register(
        "analyze_document",
        "Read and analyze the document attached to the current Telegram message or the message being replied to. Do not invent file IDs or filenames.",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 1500,
                    "description": "What to extract, explain, solve, or summarize from the document."
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        timeout=120,
    )
    async def _analyze_document(prompt: str = "Summarize the document and extract the most important information."):
        if agent.gemini is None or genai_types is None:
            return "Document analysis is unavailable because GEMINI_API_KEY is not configured."
        if not hasattr(agent, "bot"):
            return "Document analysis error: Telegram Bot instance is not connected."

        try:
            message = _resolve_media_message(agent)
            meta = _media_descriptor(message)
            if meta["kind"] != "document":
                return "Document analysis error: the current/replied message is not a document."

            tg_file = await agent.bot.get_file(meta["file_id"])
            doc_bytes = bytes(await tg_file.download_as_bytearray())
            if len(doc_bytes) > 15 * 1024 * 1024:
                return "Document analysis error: document is larger than 15 MB."

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
            return f"Document processing error: {type(exc).__name__}: {exc}"

    @r.register(
        "inspect_telegram_context",
        "Inspect Telegram metadata for the user's latest incoming message. Extracts user profile details, chat IDs, whether the message was forwarded, and the original sender's ID, name, or channel username if forwarded.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    async def _inspect_telegram_context():
        # Requires current Telegram update/message context attached to agent
        msg = getattr(agent, "current_message", None)
        if not msg:
            return json.dumps({
                "telegram_id": agent.telegram_id,
                "chat_id": agent.chat_id,
                "note": "Raw message object not cached; showing active session IDs."
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
                "title": getattr(msg.chat, "title", None)
            },
            "message_id": msg.message_id,
            "is_forwarded": bool(
                getattr(msg, "forward_date", None) 
                or getattr(msg, "forward_origin", None)
            ),
            "forward_origin": None
        }

        # Handle modern PTB v20+ forward_origin structure
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


    @r.register(
        "send_inline_keyboard",
        "Send an interactive message with clickable inline buttons. USE SPARINGLY and only when Senpai requires quick links, confirmations (Yes/No), or selection choices. Do NOT spam buttons on ordinary chats.",
        {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Message text above the buttons (Telegram Rich Markdown). Do not use HTML tags unless required by Telegram Rich Markdown."
                },
                "buttons": {
                    "type": "array",
                    "description": "2D list representing rows of buttons. Each button must have 'text' and either 'url' or 'callback_data'.",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "url": {"type": "string"},
                                "callback_data": {"type": "string"}
                            },
                            "required": ["text"]
                        }
                    }
                }
            },
            "required": ["text", "buttons"]
        },
        side_effect=True,
    )
    async def _send_inline_keyboard(text: str, buttons: list[list[dict]]):
        if not hasattr(agent, "bot"):
            return "Error: Telegram Bot instance not connected."
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            keyboard = []
            for row in buttons:
                kb_row = []
                for b in row:
                    url = b.get("url")
                    callback_data = b.get("callback_data")
                    if not url and not callback_data:
                        raise ValueError(
                            "Every inline button needs either a URL or callback_data."
                        )
                    if url and callback_data:
                        raise ValueError(
                            "A button cannot contain both URL and callback_data."
                        )
                    kb_row.append(
                        InlineKeyboardButton(
                            text=b["text"],
                            url=url,
                            callback_data=callback_data,
                        )
                    )
                keyboard.append(kb_row)

            reply_markup = InlineKeyboardMarkup(keyboard)
            await agent.bot.do_api_request(
                "sendRichMessage",
                api_kwargs={
                    "chat_id": agent.chat_id,
                    "rich_message": {"markdown": text},
                    "reply_markup": reply_markup.to_dict(),
                },
            )
            return "Success: Inline keyboard message delivered to chat."
        except Exception as exc:
            return f"Failed to send inline keyboard: {exc}"