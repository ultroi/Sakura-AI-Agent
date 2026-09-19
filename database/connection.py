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
            connectTimeoutMS=20_000,
            serverSelectionTimeoutMS=15_000,
            retryWrites=True,
            retryReads=True,
        )
        self.db = self.client[settings.mongodb_db]

    async def connect(self) -> None:
        """Establishes connection and initializes the database."""
        try:
            # Force server selection and verify that the deployment is usable.
            await self.client.admin.command({"ping": 1})
            
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
        """Runs all index creations concurrently."""
        index_tasks = [
            self.db.users.create_index("telegram_id", unique=True),
            self.db.messages.create_index([("telegram_id", 1), ("created_at", -1)]),
            self.db.notes.create_index([("telegram_id", 1), ("created_at", -1)]),
            self.db.reminders.create_index([("telegram_id", 1), ("run_at", 1)]),
            self.db.reminders.create_index("status")
        ]
        
        # Execute all index creations at the same time
        await asyncio.gather(*index_tasks)
        logger.info("MongoDB indexes verified successfully (concurrently)")

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