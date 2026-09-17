# Sakura — Personal Telegram AI Agent

Sakura is a single-agent personal assistant that lives in Telegram and can chat, use tools, remember recent conversations, create reminders, save notes, search the web, check weather, calculate, translate, fetch URLs, work with Gmail/Google Calendar/GitHub, and transcribe voice messages.

## Architecture

Telegram → `agent.py` → registered tools in `tools.py` → MongoDB / external APIs.

The LLM uses Groq tool calling in a ReAct-style loop, with a maximum of 5 tool iterations per user request.

## Requirements

- Python 3.10+
- MongoDB local or MongoDB Atlas
- Telegram Bot token
- Groq API key
- Tavily API key for web search
- GitHub personal access token for GitHub actions
- Google OAuth Desktop credentials for Gmail + Calendar

All requested external services have free-access/free-tier options, but their quotas and policies can change. Check the provider's current terms before production use.

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:

```env
BOT_TOKEN=...
GROQ_API_KEY=...
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=sakura
TAVILY_API_KEY=...
GITHUB_TOKEN=...
TIMEZONE=Asia/Kolkata
DIGEST_TIME=08:00
OWNER_TELEGRAM_ID=...
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json
```

### 3. MongoDB

For local MongoDB, start the MongoDB service. For Atlas, put the Atlas URI in `MONGODB_URI`.

### 4. Google OAuth

Create a Google Cloud project, enable Gmail API + Calendar API, configure the OAuth consent screen, create a Desktop OAuth client, and download the client JSON as `credentials.json`.

Run Sakura, then use `/connect_google` in Telegram. The standard local-browser OAuth flow will create `token.json`.

The application requests Gmail modify access and Calendar access because it must read/send email and create/read calendar events.

### 5. Run

```bash
python app.py
```

Open Telegram and send `/start`.

## Examples

- `remind me tomorrow at 10am to call John`
- `remember that my database exam is on Friday`
- `what did I save about MongoDB?`
- `what is the weather in Jaipur?`
- `search the web for the latest Python release`
- `calculate (25 * 17) / 3`
- `translate “How are you?” to Japanese`
- `summarize https://example.com/article`
- `show my upcoming calendar events`
- `send an email to ...`
- `list my GitHub repos`
- `create a GitHub issue in owner/repo ...`
- Send a Telegram voice message and Sakura will transcribe it before handling it.

## Extending Sakura with a new tool

All agent tools are registered in `tools.py` through `ToolRegistry.register`.

Add another function like:

```python
@r.register(
    "my_tool",
    "What this tool does",
    {
        "type": "object",
        "properties": {
            "value": {"type": "string"}
        },
        "required": ["value"]
    },
)
async def _my_tool(value: str):
    try:
        return f"Result: {value}"
    except Exception as exc:
        return f"Tool error: {exc}"
```

The LLM sees the schema automatically and can select the tool.

## Security notes

- Never commit `.env`, `credentials.json`, or `token.json`.
- GitHub write access and Gmail send access are powerful. Review tool calls and scopes carefully.
- This project is designed as a personal assistant, not as a public multi-tenant SaaS.
- URL fetching is intended for public URLs. Do not point it at internal services.

## Important implementation note

Reminders are persisted in MongoDB and restored on startup. Morning digest uses the Telegram job queue. If the bot is offline when a reminder becomes due, Sakura sends it as soon as possible after restart.
