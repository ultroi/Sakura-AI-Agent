from __future__ import annotations

import ast
import asyncio
import json
import math
import operator as op
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup


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
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    return str(result)

# ---------------------------------------------------------------------------
# Network error handling (Using shared agent.http_client for efficiency)
# ---------------------------------------------------------------------------
async def _http_get_json(agent, url: str, params: dict | None = None, headers: dict | None = None):
    try:
        response = await agent.http_client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise Exception("The API request timed out. The service might be slow.")
    except httpx.HTTPStatusError as exc:
        raise Exception(f"API returned an error code: {exc.response.status_code}")

async def _http_post_json(agent, url: str, payload: dict, headers: dict | None = None):
    try:
        response = await agent.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise Exception("The API request timed out.")
    except httpx.HTTPStatusError as exc:
        raise Exception(f"API returned an error code: {exc.response.status_code}")


def _clean_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())[:max_chars]


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
        "Create a Telegram reminder (a push notification ping). USE THIS FOR: 'Remind me to...', alarms, or quick personal pings. DO NOT use this for events/meetings. run_at must be an ISO-8601 timestamp in the configured local timezone.",
        {"type": "object", "properties": {"text": {"type": "string"}, "run_at": {"type": "string", "description": "ISO-8601 datetime"}}, "required": ["text", "run_at"]},
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
                agent.schedule_reminder(reminder_id, agent.chat_id, text, when)
            return f"Success! Telegram reminder created for {when.strftime('%d %b %Y at %I:%M %p %Z')}."
        except Exception as exc:
            return f"Reminder error: Could not parse time. Ensure ISO-8601 format. Details: {exc}"


    @r.register(
        "web_search",
        "Search the live web using Tavily. USE THIS for current news, facts, or things you do not confidently know.",
        {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 8}}, "required": ["query"]},
    )
    async def _web_search(query: str, max_results: int = 5):
        key = agent.settings.tavily_api_key
        if not key:
            return "Web search is unavailable because TAVILY_API_KEY is not configured."
        try:
            data = await _http_post_json(
                agent,
                "https://api.tavily.com/search",
                {"api_key": key, "query": query, "search_depth": "basic", "max_results": max(1, min(max_results, 8)), "include_answer": True},
            )
            results = [{"title": x.get("title"), "url": x.get("url"), "content": x.get("content", "")} for x in data.get("results", [])]
            return json.dumps({"answer": data.get("answer"), "results": results}, ensure_ascii=False)
        except Exception as exc:
            return f"Web search error: {exc}. Tell the user the search engine is temporarily down."

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
            response = await asyncio.to_thread(
                agent.groq.chat.completions.create,
                model="openai/gpt-oss-20b",
                messages=[{"role": "system", "content": f"Translate the user's text into {target_language}. Return only the translation."}, {"role": "user", "content": text}],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return f"Translation error: {exc}"

    @r.register(
        "fetch_url",
        "Fetch a public HTTP/HTTPS URL and extract readable text. Useful for summarizing links the user sends you.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    )
    async def _fetch_url(url: str):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return "URL error: only public HTTP/HTTPS URLs are supported."
            
            response = await agent.http_client.get(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                text = _clean_text(response.text)
            else:
                text = response.text[:15000]
            return json.dumps({"url": url, "content_type": content_type, "text": text}, ensure_ascii=False)
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
        "Generate a static satellite or roadmap image URL for a given location or address with an optional pinpoint marker.",
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Address or location (e.g. 'Bhiwadi, Rajasthan')"},
                "zoom": {"type": "integer", "description": "Zoom level between 1 (world) and 20 (building). Default 14.", "default": 14},
                "maptype": {"type": "string", "enum": ["roadmap", "satellite", "terrain", "hybrid"], "default": "roadmap"}
            },
            "required": ["location"],
        },
    )
    async def _get_map_image(location: str, zoom: int = 14, maptype: str = "roadmap"):
        api_key = getattr(agent.settings, "google_maps_api_key", None)
        if not api_key:
            return "Maps error: GOOGLE_MAPS_API_KEY is not configured in settings."
        try:
            safe_loc = quote(location)
            map_url = (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={safe_loc}&zoom={zoom}&size=600x350&scale=2&maptype={maptype}"
                f"&markers=color:red%7Clabel:S%7C{safe_loc}&key={api_key}"
            )
            return json.dumps({
                "location": location,
                "map_image_url": map_url,
                "note": "Include this URL as an HTML link or photo so the user can see the map view."
            }, ensure_ascii=False)
        except Exception as exc:
            return f"Map image generation error: {exc}"

    @r.register(
        "get_directions",
        "Calculate transit routes, travel duration, distance, and step-by-step navigation directions between two locations.",
        {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting address or landmark"},
                "destination": {"type": "string", "description": "Ending address or landmark"},
                "mode": {"type": "string", "enum": ["driving", "walking", "bicycling", "transit"], "default": "driving"}
            },
            "required": ["origin", "destination"],
        },
    )
    async def _get_directions(origin: str, destination: str, mode: str = "driving"):
        api_key = getattr(agent.settings, "google_maps_api_key", None)
        if not api_key:
            return "Maps error: GOOGLE_MAPS_API_KEY is not configured in settings."
        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "key": api_key,
            }
            data = await _http_get_json(agent, url, params=params)
            if data.get("status") != "OK":
                return f"Directions lookup failed: {data.get('status')} - {data.get('error_message', 'No route found.')}"

            route = data["routes"][0]["legs"][0]
            steps = [
                re.sub(r"<[^>]+>", "", step.get("html_instructions", ""))
                for step in route.get("steps", [])[:8]
            ]

            result = {
                "start_address": route.get("start_address"),
                "end_address": route.get("end_address"),
                "distance": route.get("distance", {}).get("text"),
                "duration": route.get("duration", {}).get("text"),
                "key_steps": steps,
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return f"Directions error: {exc}"

    @r.register(
        "search_places",
        "Search for nearby places, landmarks, restaurants, hospitals, or amenities around a location.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search target (e.g. 'coffee shops near Sohna' or 'hospitals in Bhiwadi')"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"],
        },
    )
    async def _search_places(query: str, max_results: int = 5):
        api_key = getattr(agent.settings, "google_maps_api_key", None)
        if not api_key:
            return "Maps error: GOOGLE_MAPS_API_KEY is not configured in settings."
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                "query": query,
                "key": api_key,
            }
            data = await _http_get_json(agent, url, params=params)
            if data.get("status") not in ("OK", "ZERO_RESULTS"):
                return f"Place search failed: {data.get('status')} - {data.get('error_message', '')}"

            places = []
            for item in data.get("results", [])[:max(1, min(max_results, 10))]:
                places.append({
                    "name": item.get("name"),
                    "address": item.get("formatted_address"),
                    "rating": item.get("rating", "N/A"),
                    "user_ratings_total": item.get("user_ratings_total", 0),
                    "open_now": item.get("opening_hours", {}).get("open_now", "Unknown"),
                })
            return json.dumps(places, ensure_ascii=False)
        except Exception as exc:
            return f"Places error: {exc}"

    @r.register(
        "get_map_link",
        "Generate a direct, clickable official Google Maps link for directions or place inspection on a mobile/desktop browser.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Address, coordinates, or search query"}
            },
            "required": ["query"],
        },
    )
    async def _get_map_link(query: str):
        safe_q = quote(query)
        return json.dumps({
            "query": query,
            "url": f"https://www.google.com/maps/search/?api=1&query={safe_q}"
        }, ensure_ascii=False)

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
    )
    async def _update_note(note_id: str, title: str, content: str):
        success = await agent.notes.update(note_id, agent.telegram_id, title, content)
        return f"Successfully updated the memory." if success else "Failed to update: Invalid ID or note not found."

    @r.register(
        "delete_note",
        "Delete a saved note or memory. You MUST get the note 'id' from search_notes or recent_notes first.",
        {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]},
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
    )
    async def _delete_reminder(reminder_id: str):
        # 1. Delete from database
        success = await agent.reminders.delete(reminder_id, agent.telegram_id)
        
        if success:
            # 2. Kill the background running process
            if hasattr(agent, "cancel_reminder"):
                agent.cancel_reminder(reminder_id)
            return "Successfully canceled the reminder and stopped the background process."
            
        return "Failed to cancel: Invalid ID or reminder already sent."


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
        }
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
        "Use this tool to 'see' and read an image ONLY IF Senpai explicitly asks a question about its contents. DO NOT use this if Senpai just wants to save the image.",
        {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "prompt": {"type": "string", "description": "What to ask the vision model about the image (default: 'Describe this image in detail and read any text in it.')"}
            },
            "required": ["file_id"]
        },
    )
    async def _analyze_image(file_id: str, prompt: str = "Describe this image in detail and read any text in it."):
        api_key = getattr(agent.settings, "gemini_api_key", None)
        if not api_key:
            return "Vision error: GEMINI_API_KEY is not configured in settings."
        
        try:
            import base64
            if not hasattr(agent, "bot"):
                return "Error: Telegram bot not connected."
            
            file_id = file_id.strip()
            
            # --- STEP 1: Download from Telegram ---
            try:
                tg_file = await agent.bot.get_file(file_id)
                image_bytes = await tg_file.download_as_bytearray()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
            except Exception as e:
                return f"Telegram File Download Error: Could not fetch image from Telegram. Details: {e}"
            
            # --- STEP 2: Send to Gemini ---
            # Updated to match the latest gemini-3.8-flash model
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                    ]
                }]
            }
            
            response = await agent.http_client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return f"Gemini API Error {response.status_code}: {response.text}"
                
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as exc:
            return f"Vision processing error: {type(exc).__name__}: {exc}"

    @r.register(
        "analyze_document",
        "Use this tool to read, parse, and analyze an uploaded document (PDF, Word doc, Excel/CSV, or text file) ONLY IF Senpai explicitly asks a question or instructs you to read its contents. DO NOT use this if Senpai just asks to save or store the document.",
        {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "The Telegram file_id of the document."},
                "file_name": {"type": "string", "description": "The filename with extension (e.g. report.pdf, data.xlsx)."},
                "prompt": {"type": "string", "description": "What specific information or analysis to extract from the document."}
            },
            # FIX: Made file_name required so mimetypes can accurately guess the file format.
            "required": ["file_id", "file_name"]
        },
    )
    async def _analyze_document(file_id: str, file_name: str, prompt: str = "Summarize and extract key information from this document."):
        api_key = getattr(agent.settings, "gemini_api_key", None)
        if not api_key:
            return "Document analysis error: GEMINI_API_KEY is not configured in settings."
        if not hasattr(agent, "bot"):
            return "Error: Telegram bot not connected."

        try:
            import base64
            import mimetypes

            file_id = file_id.strip()

            # 1. Determine MIME type accurately
            mime_type = "text/plain" # Default to text/plain instead of pdf for safety with txt/code files
            if file_name:
                file_name = file_name.lower()
                guessed, _ = mimetypes.guess_type(file_name)
                if guessed:
                    mime_type = guessed
                elif file_name.endswith((".xlsx", ".xls")):
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif file_name.endswith((".docx", ".doc")):
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif file_name.endswith(".csv"):
                    mime_type = "text/csv"
                elif file_name.endswith(".pdf"):
                    mime_type = "application/pdf"

            # 2. Download from Telegram
            tg_file = await agent.bot.get_file(file_id)
            doc_bytes = await tg_file.download_as_bytearray()
            base64_doc = base64.b64encode(doc_bytes).decode("utf-8")

            # 3. Request Gemini Multimodal Engine
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": base64_doc}}
                    ]
                }]
            }

            response = await agent.http_client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120.0
            )

            if response.status_code != 200:
                return f"Gemini Document API Error {response.status_code}: {response.text}"

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates or "content" not in candidates[0]:
                return "The document could not be analyzed (possibly blocked by safety filters or empty content)."

            return candidates[0]["content"]["parts"][0]["text"]

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
        }
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
                    kb_row.append(
                        InlineKeyboardButton(
                            text=b["text"],
                            url=b.get("url"),
                            callback_data=b.get("callback_data")
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