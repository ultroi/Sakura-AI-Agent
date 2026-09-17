from __future__ import annotations

import logging

from pymongo import AsyncMongoClient
from pymongo.errors import ServerSelectionTimeoutError

from config import Settings


class MongoDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncMongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db]

    async def connect(self) -> None:
        try:
            await self.client.admin.command("ping")
        except ServerSelectionTimeoutError as exc:
            logging.getLogger(__name__).error(
                "MongoDB server selection failed: %s", exc
            )
            raise

        await self.db.users.create_index("telegram_id", unique=True)
        await self.db.messages.create_index([("telegram_id", 1), ("created_at", -1)])
        await self.db.notes.create_index([("telegram_id", 1), ("created_at", -1)])
        await self.db.reminders.create_index([("telegram_id", 1), ("run_at", 1)])
        await self.db.reminders.create_index("status")

    async def close(self) -> None:
        await self.client.close()
