from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId


class UserRepository:
    def __init__(self, db):
        self.collection = db.users

    async def upsert_user(self, telegram_id: int, username: str | None, first_name: str | None):
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {
                "$set": {
                    "telegram_id": telegram_id,
                    "username": username,
                    "first_name": first_name,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def set_owner(self, telegram_id: int):
        await self.collection.update_one(
            {"telegram_id": telegram_id}, {"$set": {"is_owner": True}}, upsert=True
        )

    async def get_owner_id(self) -> int | None:
        doc = await self.collection.find_one({"is_owner": True})
        return int(doc["telegram_id"]) if doc else None

    async def get_profile(self, telegram_id: int) -> dict[str, Any]:
        """Fetch saved user context. Structured fields only — no free-text blob anymore."""
        doc = await self.collection.find_one({"telegram_id": telegram_id})
        return doc.get("profile", {}) if doc else {}

    async def update_profile(self, telegram_id: int, updates: dict[str, Any]):
        """
        Merge-safe profile update. Each call only ever touches the specific fields
        passed in — every other saved field is left completely alone, because each
        fact (name, date_of_birth, education...) now lives in its own field instead
        of one shared text blob. Empty/None values are ignored so a field can never
        be accidentally wiped by an empty string.
        """
        clean_updates = {k: v for k, v in updates.items() if v not in (None, "")}
        if not clean_updates:
            return
        flat_updates = {f"profile.{k}": v for k, v in clean_updates.items()}
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {"$set": flat_updates},
            upsert=True,
        )


class ConversationRepository:
    def __init__(self, db):
        self.collection = db.messages

    async def add(self, telegram_id: int, role: str, content: str):
        await self.collection.insert_one(
            {
                "telegram_id": telegram_id,
                "role": role,
                "content": content,
                "created_at": datetime.now(timezone.utc),
            }
        )

    async def recent(self, telegram_id: int, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.collection.find({"telegram_id": telegram_id}).sort("created_at", -1).limit(limit)
        docs = [doc async for doc in cursor]
        docs.reverse()
        return [{"role": d["role"], "content": d["content"]} for d in docs]


class NoteRepository:
    def __init__(self, db):
        self.collection = db.notes

    async def upsert_by_title(self, telegram_id: int, title: str, content: str) -> tuple[str, str]:
        """
        Smart save: if a note with the same title (case-insensitive) already exists
        for this user, UPDATE it instead of creating a duplicate. Returns
        (note_id, "created" | "updated") so the tool layer can tell Sakura which
        happened, and Sakura can tell Senpai accurately.
        """
        now = datetime.now(timezone.utc)
        existing = await self.collection.find_one(
            {
                "telegram_id": telegram_id,
                "title": {"$regex": f"^{re.escape(title.strip())}$", "$options": "i"},
            }
        )
        if existing:
            await self.collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {"content": content, "updated_at": now}},
            )
            return str(existing["_id"]), "updated"

        result = await self.collection.insert_one(
            {
                "telegram_id": telegram_id,
                "title": title.strip(),
                "content": content,
                "created_at": now,
            }
        )
        return str(result.inserted_id), "created"

    async def update(self, note_id: str, telegram_id: int, title: str, content: str) -> bool:
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(note_id), "telegram_id": telegram_id},
                {"$set": {"title": title, "content": content, "updated_at": datetime.now(timezone.utc)}},
            )
            return result.modified_count > 0
        except Exception:
            return False

    async def delete(self, note_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.delete_one(
                {"_id": ObjectId(note_id), "telegram_id": telegram_id}
            )
            return result.deleted_count > 0
        except Exception:
            return False

    async def search(self, telegram_id: int, query: str, limit: int = 10) -> list[dict[str, Any]]:
        regex = {"$regex": re.escape(query), "$options": "i"}
        cursor = self.collection.find(
            {"telegram_id": telegram_id, "$or": [{"title": regex}, {"content": regex}]}
        ).sort("created_at", -1).limit(limit)
        return [doc async for doc in cursor]

    async def recent(self, telegram_id: int, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.collection.find({"telegram_id": telegram_id}).sort("created_at", -1).limit(limit)
        return [doc async for doc in cursor]


class ReminderRepository:
    def __init__(self, db):
        self.collection = db.reminders

    async def create(self, telegram_id: int, chat_id: int, text: str, run_at: datetime) -> str:
        result = await self.collection.insert_one(
            {
                "telegram_id": telegram_id,
                "chat_id": chat_id,
                "text": text,
                "run_at": run_at,
                "status": "pending",
                "created_at": datetime.now(timezone.utc),
            }
        )
        return str(result.inserted_id)

    async def due_or_pending(self) -> list[dict[str, Any]]:
        cursor = self.collection.find({"status": "pending"}).sort("run_at", 1)
        return [doc async for doc in cursor]

    async def mark_sent(self, reminder_id: str):
        await self.collection.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": {"status": "sent", "sent_at": datetime.now(timezone.utc)}},
        )

    async def get_user_pending(self, telegram_id: int) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"telegram_id": telegram_id, "status": "pending"}
        ).sort("run_at", 1)
        return [doc async for doc in cursor]

    async def delete(self, reminder_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.delete_one(
                {"_id": ObjectId(reminder_id), "telegram_id": telegram_id, "status": "pending"}
            )
            return result.deleted_count > 0
        except Exception:
            return False