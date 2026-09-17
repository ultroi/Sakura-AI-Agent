from __future__ import annotations

import os
from email.message import EmailMessage
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
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

    def _get_service(self, name: str, version: str):
        key = f"{name}:{version}"
        if key not in self._services:
            self._services[key] = build(name, version, credentials=self._creds(), cache_discovery=False)
        return self._services[key]

    def _creds(self) -> Credentials:
        # 1. Force the file path to be the actual filename from settings
        token_file = Path(self.settings.google_token_file)
        
        # Safety Check: If someone accidentally put JSON in the .env variable, reset it
        if "{" in self.settings.google_token_file or len(self.settings.google_token_file) > 20:
            token_file = Path("token.json")
            
        creds = None
        
        # If the file exists, try to load it
        if token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
            except Exception as e:
                print(f"Warning: Corrupted token.json file. Deleting it. Error: {e}")
                token_file.unlink() # Delete bad file
                creds = None
            
            # CRITICAL FIX for scope changes: 
            if creds and set(creds.scopes) != set(SCOPES):
                print("Scopes have changed! Forcing re-authentication...")
                creds = None 

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    # If refresh fails, wipe it out and start over
                    creds = None

            if not creds:
                credentials_file = Path(self.settings.google_credentials_file)
                if not credentials_file.exists():
                    raise FileNotFoundError(
                        f"Google OAuth credentials not found: {credentials_file}. "
                        "Download a Desktop OAuth client JSON and place it there."
                    )
                
                # Run local server to get the new credentials
                flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
                creds = flow.run_local_server(port=0)
                
            # Save the new token with all 4 scopes!
            token_file.write_text(creds.to_json(), encoding="utf-8")
            
        return creds

    # --- GMAIL & CALENDAR (Existing) ---
    def gmail_list(self, query: str = "", max_results: int = 10) -> list[dict]:
        service = build("gmail", "v1", credentials=self._creds(), cache_discovery=False)
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
                    data = body["data"] + "=" * (4 - len(body["data"]) % 4)
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
        padded_data = file_data + '=' * (4 - len(file_data) % 4)
        return base64.urlsafe_b64decode(padded_data)

    def gmail_send(self, to: str, subject: str, body: str) -> dict:
        service = build("gmail", "v1", credentials=self._creds(), cache_discovery=False)
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to
        message["Subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"sent": True, "message_id": result.get("id"), "to": to, "subject": subject}

    def calendar_list(self, days: int = 7, max_results: int = 20) -> list[dict]:
        service = build("calendar", "v3", credentials=self._creds(), cache_discovery=False)
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
        service = build("calendar", "v3", credentials=self._creds(), cache_discovery=False)
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
        service = build("docs", "v1", credentials=self._creds(), cache_discovery=False)
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
        service = build("drive", "v3", credentials=self._creds(), cache_discovery=False)
        results = service.files().list(
            q=query, pageSize=max_results, fields="files(id, name, mimeType, webViewLink)"
        ).execute()
        return results.get("files", [])

    # --- NEW: MAPS STATIC URL GENERATOR ---
    def get_static_map_url(self, center: str, zoom: int = 14, size: str = "600x300") -> str:
        # Uses Google Maps Static API endpoint with your project API key
        api_key = self.settings.google_maps_api_key if hasattr(self.settings, "google_maps_api_key") else self.settings.groq_api_key # fallback or use separate key
        # Maps Static API URL format:
        return f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size={size}&maptype=roadmap&markers=color:red%7C{center}&key={self.settings.tavily_api_key}" # Note: Ensure you use your Google Maps API key in production!

    def authorize(self) -> str:
        self._creds()
        return "Google authorization completed."