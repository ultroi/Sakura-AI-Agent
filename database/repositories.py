from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from semantic_memory import SemanticNoteIndex


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

    async def clear(self, telegram_id: int):
        """Clears the recent chat history for a user to reset the context."""
        await self.collection.delete_many({"telegram_id": telegram_id})


class NoteRepository:
    def __init__(self, db, semantic_index: SemanticNoteIndex | None = None):
        self.collection = db.notes
        self.semantic_index = semantic_index or SemanticNoteIndex()

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
            await self.semantic_index.upsert(
                note_id=str(existing["_id"]),
                telegram_id=telegram_id,
                title=title.strip(),
                content=content,
            )
            return str(existing["_id"]), "updated"

        result = await self.collection.insert_one(
            {
                "telegram_id": telegram_id,
                "title": title.strip(),
                "content": content,
                "created_at": now,
                "updated_at": now,
            }
        )
        note_id = str(result.inserted_id)
        await self.semantic_index.upsert(
            note_id=note_id,
            telegram_id=telegram_id,
            title=title.strip(),
            content=content,
        )
        return note_id, "created"

    async def get(self, note_id: str, telegram_id: int) -> dict[str, Any] | None:
        try:
            return await self.collection.find_one(
                {"_id": ObjectId(note_id), "telegram_id": telegram_id}
            )
        except Exception:
            return None

    async def update(self, note_id: str, telegram_id: int, title: str, content: str) -> bool:
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(note_id), "telegram_id": telegram_id},
                {"$set": {"title": title, "content": content, "updated_at": datetime.now(timezone.utc)}},
            )
            if result.matched_count > 0:
                await self.semantic_index.upsert(
                    note_id=note_id,
                    telegram_id=telegram_id,
                    title=title,
                    content=content,
                )
            return result.modified_count > 0 or result.matched_count > 0
        except Exception:
            return False

    async def delete(self, note_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.delete_one(
                {"_id": ObjectId(note_id), "telegram_id": telegram_id}
            )
            if result.deleted_count > 0:
                await self.semantic_index.delete(note_id)
            return result.deleted_count > 0
        except Exception:
            return False

    async def search(self, telegram_id: int, query: str, limit: int = 10) -> list[dict[str, Any]]:
        regex = {"$regex": re.escape(query), "$options": "i"}
        cursor = self.collection.find(
            {"telegram_id": telegram_id, "$or": [{"title": regex}, {"content": regex}]}
        ).sort("created_at", -1).limit(limit)
        return [doc async for doc in cursor]

    async def hybrid_search(
        self,
        telegram_id: int,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Combine lexical search with optional semantic retrieval."""
        limit = max(1, min(int(limit), 20))
        lexical = await self.search(telegram_id, query, limit=limit)
        semantic = await self.semantic_index.search(
            telegram_id=telegram_id,
            query=query,
            limit=limit,
        )

        # Lazily backfill older notes into the semantic index. This avoids a
        # migration job while still allowing semantic queries to find notes that
        # were created before semantic memory was enabled.
        if self.semantic_index.enabled and not semantic:
            cursor = self.collection.find(
                {"telegram_id": telegram_id}
            ).sort("created_at", -1).limit(200)
            legacy_docs = [doc async for doc in cursor]
            for legacy in legacy_docs:
                await self.semantic_index.upsert(
                    note_id=str(legacy["_id"]),
                    telegram_id=telegram_id,
                    title=str(legacy.get("title") or ""),
                    content=str(legacy.get("content") or ""),
                )
            semantic = await self.semantic_index.search(
                telegram_id=telegram_id,
                query=query,
                limit=limit,
            )

        by_id: dict[str, dict[str, Any]] = {}
        for rank, doc in enumerate(lexical):
            item = dict(doc)
            item["_lexical_rank"] = rank
            by_id[str(doc["_id"])] = item

        for rank, note_id in enumerate(
            str(item["id"]) for item in semantic if item.get("id")
        ):
            try:
                doc = await self.collection.find_one(
                    {"_id": ObjectId(note_id), "telegram_id": telegram_id}
                )
            except Exception:
                doc = None
            if doc:
                key = str(doc["_id"])
                item = by_id.setdefault(key, dict(doc))
                item["_semantic_rank"] = rank

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in by_id.values():
            score = 0.0
            lexical_rank = item.get("_lexical_rank")
            semantic_rank = item.get("_semantic_rank")
            if lexical_rank is not None:
                score += 0.45 / (1.0 + float(lexical_rank))
            if semantic_rank is not None:
                score += 0.55 / (1.0 + float(semantic_rank))
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored[:limit]]

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

    async def update(
        self,
        reminder_id: str,
        telegram_id: int,
        *,
        text: str | None = None,
        run_at: datetime | None = None,
    ) -> bool:
        """Update the text and/or scheduled time of a user's pending reminder."""
        updates: dict[str, Any] = {}
        if text is not None:
            clean_text = text.strip()
            if not clean_text:
                return False
            updates["text"] = clean_text
        if run_at is not None:
            # Store reminder times consistently as timezone-aware UTC datetimes.
            if run_at.tzinfo is None:
                run_at = run_at.replace(tzinfo=timezone.utc)
            updates["run_at"] = run_at.astimezone(timezone.utc)

        if not updates:
            return False

        updates["updated_at"] = datetime.now(timezone.utc)
        try:
            result = await self.collection.update_one(
                {
                    "_id": ObjectId(reminder_id),
                    "telegram_id": telegram_id,
                    "status": "pending",
                },
                {"$set": updates},
            )
            return result.matched_count > 0
        except Exception:
            return False

    async def delete(self, reminder_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.delete_one(
                {"_id": ObjectId(reminder_id), "telegram_id": telegram_id, "status": "pending"}
            )
            return result.deleted_count > 0
        except Exception:
            return False

class ConversationStateRepository:
    """Persistent topic/entity continuity for natural follow-up messages."""

    def __init__(self, db):
        self.collection = db.conversation_state

    async def get(self, telegram_id: int) -> dict[str, Any]:
        doc = await self.collection.find_one({"telegram_id": telegram_id})
        if not doc:
            return {
                "telegram_id": telegram_id,
                "topic": None,
                "intent": None,
                "active_entity_type": None,
                "active_entity_id": None,
                "intent_history": [],
                "pending_confirmation": None,
            }
        doc.pop("_id", None)
        return doc

    async def set(
        self,
        telegram_id: int,
        *,
        topic: str | None = None,
        intent: str | None = None,
        active_entity_type: str | None = None,
        active_entity_id: str | None = None,
        clear_active_entity: bool = False,
        pending_confirmation: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        current = await self.get(telegram_id)
        history = list(current.get("intent_history") or [])
        if intent:
            history.append({
                "topic": topic,
                "intent": intent,
                "entity_type": active_entity_type,
                "entity_id": active_entity_id,
                "at": now.isoformat(),
            })
        updates: dict[str, Any] = {
            "telegram_id": telegram_id,
            "updated_at": now,
            "intent_history": history[-10:],
        }
        if topic is not None:
            updates["topic"] = topic
        if intent is not None:
            updates["intent"] = intent
        if clear_active_entity:
            updates["active_entity_type"] = None
            updates["active_entity_id"] = None
        else:
            if active_entity_type is not None:
                updates["active_entity_type"] = active_entity_type
            if active_entity_id is not None:
                updates["active_entity_id"] = active_entity_id
        if pending_confirmation is not None:
            updates["pending_confirmation"] = pending_confirmation
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {"$set": updates, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    async def clear_confirmation(self, telegram_id: int) -> None:
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {"$set": {
                "pending_confirmation": None,
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )


class WorkflowStateRepository:
    """First-class workflow state independent of callback_data strings."""

    def __init__(self, db):
        self.collection = db.telegram_workflows

    async def get(
        self,
        telegram_id: int,
        chat_id: int,
        workflow_id: str,
    ) -> dict[str, Any] | None:
        return await self.collection.find_one({
            "telegram_id": telegram_id,
            "chat_id": chat_id,
            "workflow_id": workflow_id,
        })

    async def set(
        self,
        telegram_id: int,
        chat_id: int,
        workflow_id: str,
        workflow_type: str,
        state: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {
                "telegram_id": telegram_id,
                "chat_id": chat_id,
                "workflow_id": workflow_id,
            },
            {
                "$set": {
                    "telegram_id": telegram_id,
                    "chat_id": chat_id,
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "state": state,
                    "data": data or {},
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def clear(self, telegram_id: int, chat_id: int, workflow_id: str) -> bool:
        result = await self.collection.delete_one({
            "telegram_id": telegram_id,
            "chat_id": chat_id,
            "workflow_id": workflow_id,
        })
        return result.deleted_count > 0


class PersistentEntityStore:
    """Cross-turn cache of real IDs returned by Sakura tools.

    This is intentionally separate from short-lived RequestContext so prerequisite
    checks can safely accept IDs that were produced in an earlier turn.
    """

    MAX_PER_KIND = 50

    def __init__(self, db):
        self.collection = db.sakura_entity_cache

    async def load(self, telegram_id: int) -> dict[str, list[dict[str, Any]]]:
        doc = await self.collection.find_one({"telegram_id": telegram_id})
        raw = doc.get("entities", {}) if doc else {}
        if not isinstance(raw, dict):
            return {}
        cleaned: dict[str, list[dict[str, Any]]] = {}
        for kind, items in raw.items():
            if not isinstance(items, list):
                continue
            rows: list[dict[str, Any]] = []
            for item in items[-self.MAX_PER_KIND:]:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                rows.append({
                    "id": str(item["id"]),
                    "label": str(item.get("label") or ""),
                    "tool": str(item.get("tool") or ""),
                    "updated_at": str(item.get("updated_at") or ""),
                })
            cleaned[str(kind)] = rows
        return cleaned

    async def save(self, telegram_id: int, entities: dict[str, list[dict[str, Any]]]) -> None:
        now = datetime.now(timezone.utc)
        compact: dict[str, list[dict[str, Any]]] = {}
        for kind, items in entities.items():
            dedup: dict[str, dict[str, Any]] = {}
            for item in items:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                key = str(item["id"])
                dedup[key] = {
                    "id": key,
                    "label": str(item.get("label") or ""),
                    "tool": str(item.get("tool") or ""),
                    "updated_at": str(item.get("updated_at") or now.isoformat()),
                }
            compact[str(kind)] = list(dedup.values())[-self.MAX_PER_KIND:]
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {
                "$set": {
                    "telegram_id": telegram_id,
                    "entities": compact,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )


class WatchRepository:
    """Persistent condition-watch storage for anime/product/site monitoring."""

    def __init__(self, db):
        self.collection = db.watchers

    async def create(
        self,
        telegram_id: int,
        chat_id: int,
        *,
        kind: str,
        target: str,
        condition: str,
        query: str,
        url: str,
        interval_minutes: int,
    ) -> str:
        now = datetime.now(timezone.utc)
        result = await self.collection.insert_one(
            {
                "telegram_id": telegram_id,
                "chat_id": chat_id,
                "kind": kind,
                "target": target.strip(),
                "condition": condition.strip(),
                "query": query.strip(),
                "url": url.strip(),
                "interval_minutes": int(interval_minutes),
                "enabled": True,
                "next_check_at": now,
                "last_checked_at": None,
                "last_state": None,
                "seen_keys": [],
                "last_error": None,
                "failure_count": 0,
                "last_failure_alert_at": None,
                "last_notified_at": None,
                "last_notification_key": None,
                "notification_count": 0,
                "acknowledged_at": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return str(result.inserted_id)

    async def count_user(self, telegram_id: int) -> int:
        return int(await self.collection.count_documents({"telegram_id": telegram_id, "enabled": True}))

    async def list_user(self, telegram_id: int) -> list[dict[str, Any]]:
        cursor = self.collection.find({"telegram_id": telegram_id}).sort("created_at", -1)
        return [doc async for doc in cursor]

    async def get_due(self, now: datetime, limit: int = 20) -> list[dict[str, Any]]:
        cursor = (
            self.collection.find(
                {
                    "enabled": True,
                    "next_check_at": {"$lte": now},
                }
            )
            .sort("next_check_at", 1)
            .limit(limit)
        )
        return [doc async for doc in cursor]

    async def update_state(
        self,
        watch_id: str,
        *,
        next_check_at: datetime,
        last_checked_at: datetime,
        last_state: Any = None,
        seen_keys: list[str] | None = None,
        last_error: str | None = None,
        failure_count: int | None = None,
    ) -> None:
        updates: dict[str, Any] = {
            "next_check_at": next_check_at,
            "last_checked_at": last_checked_at,
            "updated_at": datetime.now(timezone.utc),
            "last_error": last_error,
        }
        if last_state is not None:
            updates["last_state"] = last_state
        if seen_keys is not None:
            updates["seen_keys"] = seen_keys[-50:]
        if failure_count is not None:
            updates["failure_count"] = max(0, int(failure_count))
        try:
            await self.collection.update_one(
                {"_id": ObjectId(watch_id)},
                {"$set": updates},
            )
        except Exception:
            pass

    async def mark_notified(
        self,
        watch_id: str,
        *,
        notification_key: str,
        notified_at: datetime,
    ) -> None:
        try:
            await self.collection.update_one(
                {"_id": ObjectId(watch_id)},
                {
                    "$set": {
                        "last_notified_at": notified_at,
                        "last_notification_key": notification_key,
                        "acknowledged_at": None,
                        "updated_at": notified_at,
                    },
                    "$inc": {"notification_count": 1},
                },
            )
        except Exception:
            pass

    async def acknowledge_hit(self, watch_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(watch_id), "telegram_id": telegram_id},
                {"$set": {
                    "acknowledged_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }},
            )
            return result.matched_count > 0
        except Exception:
            return False
        
    async def should_notify(
        self,
        watch: dict[str, Any],
        *,
        notification_key: str,
        now: datetime,
        cooldown_minutes: int,
    ) -> bool:
        last_key = str(watch.get("last_notification_key") or "")
        if last_key != notification_key:
            return True
        last_notified = watch.get("last_notified_at")
        if not last_notified:
            return True
        if getattr(last_notified, "tzinfo", None) is None:
            last_notified = last_notified.replace(tzinfo=timezone.utc)
        return (now - last_notified.astimezone(timezone.utc)).total_seconds() >= cooldown_minutes * 60

    async def update_config(
        self,
        watch_id: str,
        telegram_id: int,
        *,
        interval_minutes: int | None = None,
        condition: str | None = None,
        query: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            oid = ObjectId(watch_id)
        except Exception:
            return None

        updates: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if interval_minutes is not None:
            interval_minutes = int(interval_minutes)
            updates["interval_minutes"] = interval_minutes
            updates["next_check_at"] = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
        if condition is not None:
            updates["condition"] = condition.strip()
        if query is not None:
            updates["query"] = query.strip()
        if url is not None:
            updates["url"] = url.strip()

        # A changed watch definition gets a fresh notification baseline so an old
        # hit cannot suppress or duplicate the next notification.
        if any(value is not None for value in (condition, query, url)):
            updates["seen_keys"] = []
            updates["last_notification_key"] = None
            updates["last_notified_at"] = None
            updates["acknowledged_at"] = None
            updates["failure_count"] = 0

        if len(updates) == 1:
            return None

        try:
            from pymongo import ReturnDocument
            return await self.collection.find_one_and_update(
                {"_id": oid, "telegram_id": telegram_id},
                {"$set": updates},
                return_document=ReturnDocument.AFTER,
            )
        except Exception:
            return None

    async def delete(self, watch_id: str, telegram_id: int) -> bool:
        try:
            result = await self.collection.delete_one(
                {"_id": ObjectId(watch_id), "telegram_id": telegram_id}
            )
            return result.deleted_count > 0
        except Exception:
            return False


class TelegramUIRepository:
    """Persistent state for dynamically managed Telegram UI/messages."""

    def __init__(self, db):
        self.collection = db.telegram_ui_state

    async def remember(
        self,
        telegram_id: int,
        chat_id: int,
        message_id: int,
        *,
        text: str = "",
        buttons: list[list[dict[str, Any]]] | None = None,
        callback_data: str | None = None,
        kind: str = "interactive",
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"telegram_id": telegram_id, "chat_id": chat_id, "message_id": int(message_id)},
            {
                "$set": {
                    "telegram_id": telegram_id,
                    "chat_id": chat_id,
                    "message_id": int(message_id),
                    "text": text or "",
                    "buttons": buttons if buttons is not None else [],
                    "callback_data": callback_data,
                    "kind": kind,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def latest(self, telegram_id: int, chat_id: int) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"telegram_id": telegram_id, "chat_id": chat_id},
            sort=[("updated_at", -1)],
        )

    async def recent(self, telegram_id: int, chat_id: int, limit: int = 5) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"telegram_id": telegram_id, "chat_id": chat_id}
        ).sort("updated_at", -1).limit(max(1, min(limit, 10)))
        return [doc async for doc in cursor]

    async def remove(self, telegram_id: int, chat_id: int, message_id: int) -> None:
        await self.collection.delete_one(
            {"telegram_id": telegram_id, "chat_id": chat_id, "message_id": int(message_id)}
        )
