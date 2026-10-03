from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ReturnDocument


# The global budget document lives in its own collection. It is scoped to the
# whole installation (i.e. the whole Groq organization), NOT to a Telegram
# user, because Groq enforces its daily quota at the organization level.
_GROQ_BUDGET_ID = "groq_daily_budget"
_DEFAULT_GLOBAL_COLLECTION = "sakura_global_state"


class SelfStateStore:
    """
    Persistent operational state for Sakura.

    User-specific state lives in `sakura_self_state`, keyed by telegram_id.
    The Groq daily token budget lives in a separate global collection, because
    Groq's rate limit is shared across every request the installation makes —
    not per Telegram user.

    Reservation flow used by the agent:

        reserved = await store.reserve_groq_tokens(estimate, daily_limit=L)
        if reserved is None: -> budget exhausted, do not call Groq
        try:
            response = await groq(...)
            actual = response.usage.total_tokens
            await store.reconcile_groq_reservation(reserved, actual)
        except Exception:
            await store.release_groq_reservation(reserved)
            raise

    This guarantees that two concurrent requests cannot both pass the budget
    check and then blow past the limit — the reservation is atomic.
    """

    def __init__(
        self,
        db,
        *,
        collection_name: str = "sakura_self_state",
        global_collection_name: str = _DEFAULT_GLOBAL_COLLECTION,
    ):
        self.collection = getattr(db, collection_name)
        self.global_collection = getattr(db, global_collection_name)

    # -----------------------------------------------------------------------
    # User-specific state (unchanged public API)
    # -----------------------------------------------------------------------
    async def get(self, telegram_id: int) -> dict[str, Any]:
        doc = await self.collection.find_one({"telegram_id": telegram_id})
        if not doc:
            return {
                "telegram_id": telegram_id,
                "last_tool_used": None,
                "last_successful_tool": None,
                "recent_failures": [],
                "active_workflow": None,
                "known_user_prefs": {},
            }
        doc.pop("_id", None)
        return doc

    async def update(self, telegram_id: int, **updates: Any) -> None:
        updates["updated_at"] = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"telegram_id": telegram_id},
            {
                "$set": {
                    "telegram_id": telegram_id,
                    **updates,
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    async def record_tool(
        self,
        telegram_id: int,
        *,
        tool_name: str,
        ok: bool,
        error: str | None = None,
    ) -> None:
        current = await self.get(telegram_id)
        failures = list(current.get("recent_failures") or [])
        if ok:
            await self.update(
                telegram_id,
                last_tool_used=tool_name,
                last_successful_tool=tool_name,
                recent_failures=failures[-4:],
            )
            return

        failures.append({
            "tool": tool_name,
            "error": (error or "")[:500],
            "at": datetime.now(timezone.utc).isoformat(),
        })
        await self.update(
            telegram_id,
            last_tool_used=tool_name,
            recent_failures=failures[-5:],
        )

    # -----------------------------------------------------------------------
    # Global Groq budget (reservation-based)
    # -----------------------------------------------------------------------
    async def reserve_groq_tokens(
        self,
        estimated_tokens: int,
        *,
        daily_limit: int,
        safety_margin: float = 0.05,
    ) -> int | None:
        """
        Atomically reserve `estimated_tokens` from today's global budget.

        Returns the reserved amount on success, or None if there is not enough
        budget left. Callers MUST later either reconcile (on success) or release
        (on failure) the reservation — never both, and never neither.
        """
        estimated_tokens = max(0, int(estimated_tokens))
        if estimated_tokens == 0:
            return 0

        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        margin = max(0.0, min(float(safety_margin), 0.5))
        soft_limit = int(daily_limit * (1.0 - margin))
        daily_limit = int(daily_limit)

        # 1. Lazy day rollover. If the stored date is stale, wipe the counters.
        #    If the doc doesn't exist yet, this is a no-op and step 2 creates it.
        await self.global_collection.update_one(
            {"_id": _GROQ_BUDGET_ID, "date": {"$ne": today}},
            {
                "$set": {
                    "date": today,
                    "used": 0,
                    "reserved": 0,
                    "limit": daily_limit,
                    "soft_limit": soft_limit,
                    "updated_at": now,
                }
            },
        )

        # 2. Ensure the global doc exists for today.
        await self.global_collection.update_one(
            {"_id": _GROQ_BUDGET_ID},
            {
                "$setOnInsert": {
                    "date": today,
                    "used": 0,
                    "reserved": 0,
                    "limit": daily_limit,
                    "soft_limit": soft_limit,
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

        # 3. Atomic reservation guarded by (used + reserved + estimate <= soft_limit).
        #    Two concurrent callers cannot both succeed past the limit because
        #    MongoDB applies the filter + update atomically on a single document.
        reserved_doc = await self.global_collection.find_one_and_update(
            {
                "_id": _GROQ_BUDGET_ID,
                "date": today,
                "$expr": {
                    "$lte": [
                        {
                            "$add": [
                                {"$ifNull": ["$used", 0]},
                                {"$ifNull": ["$reserved", 0]},
                                estimated_tokens,
                            ]
                        },
                        soft_limit,
                    ]
                },
            },
            {
                "$inc": {"reserved": estimated_tokens},
                "$set": {
                    "date": today,
                    "soft_limit": soft_limit,
                    "limit": daily_limit,
                    "updated_at": now,
                },
            },
            return_document=ReturnDocument.AFTER,
        )

        if reserved_doc is None:
            return None

        return estimated_tokens

    async def reconcile_groq_reservation(
        self,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> None:
        """
        Replace a reservation with the actual usage Groq reported.

        Decrements `reserved` by the amount we held, and increments `used`
        by the real token count. If the actual usage exceeds the estimate
        (rare), the surplus is absorbed by the counter — the next reservation
        will simply see a higher `used` value.
        """
        reserved_tokens = max(0, int(reserved_tokens))
        actual_tokens = max(0, int(actual_tokens))
        now = datetime.now(timezone.utc)

        await self.global_collection.update_one(
            {"_id": _GROQ_BUDGET_ID},
            {
                "$inc": {
                    "reserved": -reserved_tokens,
                    "used": actual_tokens,
                },
                "$set": {"updated_at": now},
            },
        )

    async def release_groq_reservation(self, reserved_tokens: int) -> None:
        """Return an unused reservation to the pool (e.g. request failed)."""
        reserved_tokens = max(0, int(reserved_tokens))
        if reserved_tokens == 0:
            return

        now = datetime.now(timezone.utc)
        await self.global_collection.update_one(
            {"_id": _GROQ_BUDGET_ID},
            {
                "$inc": {"reserved": -reserved_tokens},
                "$set": {"updated_at": now},
            },
        )

    async def get_groq_budget_status(
        self,
        *,
        daily_limit: int,
        safety_margin: float = 0.05,
    ) -> dict[str, Any]:
        """Read the current global Groq budget state (safe to call anytime)."""
        doc = await self.global_collection.find_one({"_id": _GROQ_BUDGET_ID})
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        margin = max(0.0, min(float(safety_margin), 0.5))
        soft_limit = int(daily_limit * (1.0 - margin))

        if not doc or doc.get("date") != today:
            return {
                "date": today,
                "used": 0,
                "reserved": 0,
                "limit": int(daily_limit),
                "soft_limit": soft_limit,
                "remaining": soft_limit,
                "exhausted": False,
            }

        used = int(doc.get("used") or 0)
        reserved = int(doc.get("reserved") or 0)
        committed = used + reserved
        return {
            "date": today,
            "used": used,
            "reserved": reserved,
            "limit": int(daily_limit),
            "soft_limit": soft_limit,
            "remaining": max(0, soft_limit - committed),
            "exhausted": committed >= soft_limit,
        }