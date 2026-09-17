from __future__ import annotations

from database.repositories import UserRepository


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    async def ensure_user(self, update):
        user = update.effective_user
        if not user:
            return
        await self.repo.upsert_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
        )
        # Sakura is designed as a personal assistant. If no explicit owner
        # is configured, the first user who starts the bot becomes the owner.
        if await self.repo.get_owner_id() is None:
            await self.repo.set_owner(user.id)
