from __future__ import annotations

import os
import json
from email.message import EmailMessage
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote
from typing import Any

from google.auth.transport.requests import Request
from google_auth_httplib2 import AuthorizedHttp
import httplib2
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import Settings

# EXPANDED SCOPES: Added Docs and Drive Readonly
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


class GoogleService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._services: dict[str, Any] = {}

    def _http(self):
        # googleapiclient/httplib2 otherwise has no bounded request timeout.
        return AuthorizedHttp(
            self._creds(),
            http=httplib2.Http(timeout=25),
        )

    def _get_service(self, name: str, version: str):
        key = f"{name}:{version}"
        if key not in self._services:
            self._services[key] = build(
                name,
                version,
                http=self._http(),
                cache_discovery=False,
            )
        return self._services[key]

    def _creds(self) -> Credentials:
        """
        Load Google OAuth credentials and user token.

        StackHost:
          GOOGLE_CREDENTIALS_FILE = complete OAuth client JSON
          GOOGLE_TOKEN_FILE        = complete authorized-user token JSON

        Local development can still use file paths and run the browser flow.
        """
        credentials_value = self.settings.google_credentials_file.strip()
        token_value = self.settings.google_token_file.strip()

        credentials_json = None
        token_json = None

        # Environment values may contain JSON directly or a path to a JSON file.
        if credentials_value.startswith("{"):
            try:
                credentials_json = json.loads(credentials_value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "GOOGLE_CREDENTIALS_FILE contains invalid JSON."
                ) from exc
            credentials_path = None
        else:
            credentials_path = Path(credentials_value)

        if token_value.startswith("{"):
            try:
                token_json = json.loads(token_value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "GOOGLE_TOKEN_FILE contains invalid JSON."
                ) from exc
            token_path = None
        else:
            token_path = Path(token_value)

        creds = None

        # Prefer the token supplied directly through the environment.
        if token_json is not None:
            try:
                creds = Credentials.from_authorized_user_info(
                    token_json, scopes=SCOPES
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError(
                    "GOOGLE_TOKEN_FILE is not a valid Google authorized-user token."
                ) from exc
        elif token_path and token_path.is_file():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), SCOPES
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError(
                    f"Could not read Google token file: {token_path}"
                ) from exc

        if creds and not creds.valid:
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request(timeout=20))
                except Exception as exc:
                    raise RuntimeError(
                        "Google token refresh failed. Reauthorize locally and "
                        "update GOOGLE_TOKEN_FILE with the new token JSON."
                    ) from exc
            else:
                creds = None

        if creds and creds.valid:
            # Persist refreshed tokens only when using a local token file.
            if token_json is None and token_path is not None:
                token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds

        # If JSON was supplied through StackHost environment variables, do not
        # attempt to launch a browser in the headless deployment container.
        if credentials_json is not None or token_json is not None:
            raise RuntimeError(
                "Google authorization token is missing or invalid. "
                "Generate token.json locally using the same SCOPES, then paste "
                "its complete JSON into GOOGLE_TOKEN_FILE on StackHost."
            )

        if not credentials_path or not credentials_path.is_file():
            raise FileNotFoundError(
                f"Google OAuth credentials not found: {credentials_path}. "
                "Set GOOGLE_CREDENTIALS_FILE to a JSON file path or its JSON content."
            )

        # Local-only interactive authorization flow.
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path), SCOPES
        )
        creds = flow.run_local_server(port=0)

        if token_path is not None:
            token_path.write_text(creds.to_json(), encoding="utf-8")

        return creds

    # --- GMAIL & CALENDAR (Existing) ---
    def gmail_list(self, query: str = "", max_results: int = 10) -> list[dict]:
        service = self._get_service("gmail", "v1")
        response = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        items = []
        for item in response.get("messages", []):
            msg = service.users().messages().get(userId="me", id=item["id"], format="metadata", metadataHeaders=["From", "To", "Subject", "Date"]).execute()
            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            items.append({"id": item["id"], "from": headers.get("from"), "to": headers.get("to"), "subject": headers.get("subject"), "date": headers.get("date"), "snippet": msg.get("snippet", "")})
        return items

    def gmail_read(self, message_id: str) -> dict:
        service = self._get_service("gmail", "v1")
        msg = service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        headers = {
            h["name"].lower(): h["value"]
            for h in msg.get("payload", {}).get("headers", [])
        }

        attachments = []
        body_text = ""

        def _traverse(part):
            nonlocal body_text
            filename = part.get("filename", "")
            body = part.get("body", {})
            attachment_id = body.get("attachmentId")

            # Extract attachment reference
            if filename and attachment_id:
                attachments.append({
                    "filename": filename,
                    "mimeType": part.get("mimeType", "application/octet-stream"),
                    "attachment_id": attachment_id,
                    "size": body.get("size", 0)
                })

            # Extract body text snippet
            if part.get("mimeType") == "text/plain" and "data" in body:
                try:
                    data = body["data"] + "=" * (-len(body["data"]) % 4)
                    body_text += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore") + "\n"
                except Exception:
                    pass

            for subpart in part.get("parts", []):
                _traverse(subpart)

        _traverse(msg.get("payload", {}))

        # Fallback body if plain text is empty
        if not body_text:
            body_text = msg.get("snippet", "")

        return {
            "id": message_id,
            "from": headers.get("from", "Unknown"),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", "(No Subject)"),
            "date": headers.get("date", ""),
            "body": body_text[:4000],  # Keep token count tight
            "attachments": attachments
        }

    def gmail_download_attachment(self, message_id: str, attachment_id: str) -> bytes:
        service = self._get_service("gmail", "v1")
        att = service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id
        ).execute()
        
        file_data = att["data"]
        # Add padding to ensure safe decoding
        padded_data = file_data + '=' * (-len(file_data) % 4)
        return base64.urlsafe_b64decode(padded_data)

    def gmail_send(self, to: str, subject: str, body: str) -> dict:
        service = self._get_service("gmail", "v1")
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to
        message["Subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"sent": True, "message_id": result.get("id"), "to": to, "subject": subject}

    def calendar_list(self, days: int = 7, max_results: int = 20) -> list[dict]:
        service = self._get_service("calendar", "v3")
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        events = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        ).execute().get("items", [])
        return [
            {
                "id": e.get("id"),
                "summary": e.get("summary"),
                "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date"),
                "location": e.get("location"),
            }
            for e in events
        ]

    def calendar_create(self, summary: str, start_iso: str, end_iso: str, description: str = "") -> dict:
        service = self._get_service("calendar", "v3")
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        return {"created": True, "id": created.get("id"), "html_link": created.get("htmlLink"), "summary": summary}

    # --- NEW: GOOGLE DOCS ---
    def docs_read(self, document_id: str) -> str:
        service = self._get_service("docs", "v1")
        doc = service.documents().get(documentId=document_id).execute()
        text = ""
        for content in doc.get("body", {}).get("content", []):
            if "paragraph" in content:
                for element in content["paragraph"].get("elements", []):
                    if "textRun" in element:
                        text += element["textRun"].get("content", "")
        return text

    # --- NEW: GOOGLE DRIVE ---
    def drive_list(self, query: str = "", max_results: int = 10) -> list[dict]:
        service = self._get_service("drive", "v3")
        results = service.files().list(
            q=query, pageSize=max_results, fields="files(id, name, mimeType, webViewLink)"
        ).execute()
        return results.get("files", [])

    # Google Maps API has been removed from the free-only build.
    def get_static_map_url(self, center: str, zoom: int = 14, size: str = "600x300") -> str:
        safe = quote(center)
        return f"https://www.openstreetmap.org/search?query={safe}"

    def authorize(self) -> str:
        self._creds()
        return "Google authorization completed."