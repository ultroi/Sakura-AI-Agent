# Sakura AI 🌸 — Free-tier agent runtime

This branch is a reliability/security refactor of Sakura AI. It keeps the Telegram personal-assistant behavior while making the LLM/tool loop deterministic, bounded, and easier to debug.

## Free-only provider policy

The runtime is intentionally limited to free-tier model/API paths:

- **Groq**: `openai/gpt-oss-120b` for primary text/tool orchestration.
- **Google Gemini API**: `gemini-3.8-flash` for free-tier fallback and multimodal document/image analysis.
- **DuckDuckGo HTML** for web search (no paid search API key).
- **Open-Meteo** for weather/geocoding.
- **OpenStreetMap Nominatim + OSRM** for places/directions.
- **Gmail / Calendar / Drive / Docs** through the user's Google account OAuth APIs.
- **GitHub API** for the user's authenticated GitHub account.

No OpenAI API, AWS, Tavily, Google Maps Platform, or paid MongoDB/LLM requirement is included.

> Free public services still have usage policies and rate limits. “Free” does not mean unlimited throughput.

## Runtime architecture

```text
Telegram update
      │
      ▼
PTB handler ── bounded request timeouts ──► SakuraAgent
                                               │
                           ┌───────────────────┴───────────────────┐
                           │                                       │
                      Groq GPT-OSS                         Gemini Flash fallback
                           │                                       │
                           └───────────────┬───────────────────────┘
                                           ▼
                                  Tool registry / schema gate
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
              Google/GitHub          Free web/maps/weather       Memory/reminders
                  │                        │                        │
                  └────────────────────────┴────────────────────────┘
                                           ▼
                                  Structured finalization
                                           │
                                           ▼
                                Telegram Markdown renderer
```

## Major reliability changes

1. **No synchronous Groq client inside async handlers.** The refactor uses `AsyncGroq` with explicit connection/read/write/pool limits.
2. **No fake Groq model failover on 429.** GPT-OSS 120B and 20B share the same organization-level free-plan quota, so the runtime treats a Groq 429 as a provider-level failure and moves to Gemini instead of immediately retrying the same quota wall.
3. **Task-local Telegram context.** User/chat/message state is stored in `contextvars`, so concurrent updates cannot overwrite each other's tool target.
4. **Tool routing is query-aware and capped.** The model sees only a small subset of relevant tools; parallel tool execution is disabled for deterministic stateful workflows.
5. **Arguments are JSON-schema validated before execution.** Invalid arguments become structured tool errors instead of Python `TypeError` strings.
6. **Lookup IDs have provenance.** Gmail reads, note updates/deletes, attachment sends, and reminder deletes can only use IDs returned by prerequisite tools during the current request.
7. **Side-effect guard.** Tools that send mail, create events/issues, mutate memory, connect OAuth, or send Telegram media/buttons are blocked unless the user message clearly expresses that intent.
8. **Model output is Markdown-only.** Model-generated HTML is stripped; regular Telegram fallback uses a deterministic Markdown→HTML renderer.
9. **Strict finalization.** Groq structured output is used only for the final, tool-free answer because Groq currently does not support structured outputs and tool use in the same request.
10. **Prompt-injection hardening.** Saved profile values, notes, web pages, files, emails, and tool results are explicitly data, not instructions.
11. **SSRF protection.** URL-fetching blocks loopback/private/link-local/reserved addresses and validates redirect targets.
12. **Telegram traffic is reduced.** Thinking-message animation updates every 4 seconds instead of repeatedly hammering Telegram roughly every second.
13. **Voice routing is fixed.** Voice messages go to `voice_handler`; they no longer fall through to `text_handler`.
14. **Google/GitHub/Mongo requests are bounded.** Google uses a finite `httplib2` timeout, GitHub has a 20s client timeout, and MongoDB has bounded connect/server-selection/socket timeouts.

## Configuration

Copy `.env.example` to `.env` and fill only the keys you actually use.

For Google OAuth, place `credentials.json` and (after authorization) `token.json` beside the application, or inject them through the deployment environment. They are ignored by git.

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Smoke validation

```bash
python -m py_compile $(find . -name '*.py' -not -path './node_modules/*')
python tests/smoke_static.py
```

## Important security action before deployment

The supplied archive contained credential-bearing files. **Rotate/revoke those credentials before deploying this cleaned tree**. Do not copy the original `.env`, OAuth token, credentials file, bot token, Groq key, Gemini key, Mongo credentials, or GitHub token back into source control.

## Current vendor references

- Groq rate limits: https://console.groq.com/docs/rate-limits
- Groq local tool calling: https://console.groq.com/docs/tool-use/local-tool-calling
- Groq structured outputs: https://console.groq.com/docs/structured-outputs
- Gemini pricing/free tier: https://ai.google.dev/gemini-api/docs/pricing
- Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
- python-telegram-bot application/timeouts: https://docs.python-telegram-bot.org/en/latest/telegram.ext.application.html
