from __future__ import annotations

import asyncio
import logging
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.errors import (
    PyMongoError,
    ServerSelectionTimeoutError,
)

logger = logging.getLogger(__name__)


class MongoDatabase:
    # Use your actual Settings class type instead of Any if you have one (e.g., from pydantic)
    def __init__(self, settings: Any):
        self.settings = settings

        self.client = AsyncMongoClient(
            settings.mongodb_uri,
            connectTimeoutMS=6_000,
            serverSelectionTimeoutMS=8_000,
            socketTimeoutMS=20_000,
            retryWrites=True,
            retryReads=True,
            maxPoolSize=20,
            minPoolSize=1,
            maxIdleTimeMS=60_000,
        )
        self.db = self.client[settings.mongodb_db]

    async def connect(self) -> None:
        """Establishes connection and initializes the database."""
        try:
            # Force server selection once, with a hard application-level timeout.
            await asyncio.wait_for(
                self.client.admin.command({"ping": 1}),
                timeout=10,
            )
            
            logger.info(
                "MongoDB connected successfully | database=%s",
                self.settings.mongodb_db,
            )

            # Delegate index creation to a separate method
            await self._setup_indexes()

        except ServerSelectionTimeoutError as exc:
            logger.error(
                "MongoDB server selection failed. "
                "The client could not find a usable PRIMARY.\n"
                "Check MongoDB Atlas cluster health, replica-set election, "
                "and network connectivity.\n"
                "Details: %s",
                exc,
            )
            raise
        except PyMongoError as exc:
            logger.exception("MongoDB connection/initialization failed: %s", exc)
            raise

    async def _setup_indexes(self) -> None:
        """Create indexes sequentially to avoid startup connection bursts on small tiers."""
        index_specs = [
            (self.db.users, "telegram_id", {"unique": True}),
            (self.db.messages, [("telegram_id", 1), ("created_at", -1)], {}),
            (self.db.notes, [("telegram_id", 1), ("created_at", -1)], {}),
            (self.db.reminders, [("telegram_id", 1), ("run_at", 1)], {}),
            (self.db.reminders, "status", {}),
        ]
        for collection, keys, options in index_specs:
            await asyncio.wait_for(
                collection.create_index(keys, **options),
                timeout=8,
            )
        logger.info("MongoDB indexes verified successfully")

    async def close(self) -> None:
        """Closes the database connection."""
        await self.client.close()
        logger.info("MongoDB connection closed")

    # Optional: Allows using `async with MongoDatabase(settings) as db:`
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()