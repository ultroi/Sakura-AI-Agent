from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError
import logging

logger = logging.getLogger(__name__)


class MongoDatabase:
    def __init__(self, settings):
        self.settings = settings

        self.client = AsyncMongoClient(
            settings.mongodb_uri,
            connectTimeoutMS=20000,
            serverSelectionTimeoutMS=30000,
        )

        self.db = self.client[settings.mongodb_db]

    async def connect(self) -> None:
        try:
            await self.client.admin.command("ping")
            logger.info("MongoDB connected successfully")

            await self.db.users.create_index(
                "telegram_id",
                unique=True
            )

            await self.db.messages.create_index(
                [("telegram_id", 1), ("created_at", -1)]
            )

            await self.db.notes.create_index(
                [("telegram_id", 1), ("created_at", -1)]
            )

            await self.db.reminders.create_index(
                [("telegram_id", 1), ("run_at", 1)]
            )

            await self.db.reminders.create_index("status")

        except PyMongoError:
            logger.exception("MongoDB connection failed")
            raise

    async def close(self) -> None:
        await self.client.close()