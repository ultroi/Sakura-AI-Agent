from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

try:
    import chromadb  # type: ignore
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


class SemanticNoteIndex:
    """
    Optional persistent semantic index for notes.

    It activates only when both ChromaDB and sentence-transformers are installed.
    The rest of Sakura continues to work normally when they are unavailable.
    """

    def __init__(self, collection_name: str = "sakura_notes"):
        self.enabled = chromadb is not None and SentenceTransformer is not None
        self.model_name = os.getenv(
            "SAKURA_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )
        self.persist_path = os.getenv(
            "SAKURA_CHROMA_PATH",
            str(Path("data") / "chroma"),
        )
        self.collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)[:63]
        self._client = None
        self._collection = None
        self._model = None
        self._init_lock = asyncio.Lock()

    async def _ensure_ready(self) -> bool:
        if not self.enabled:
            return False

        async with self._init_lock:
            if self._collection is not None and self._model is not None:
                return True

            def _init():
                Path(self.persist_path).mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=self.persist_path)
                collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                model = SentenceTransformer(self.model_name)
                return client, collection, model

            try:
                self._client, self._collection, self._model = await asyncio.to_thread(_init)
                return True
            except Exception:
                self._client = None
                self._collection = None
                self._model = None
                self.enabled = False
                return False

    @staticmethod
    def _document_text(title: str, content: str) -> str:
        return f"Title: {title}\nContent: {content}".strip()

    async def upsert(
        self,
        *,
        note_id: str,
        telegram_id: int,
        title: str,
        content: str,
    ) -> bool:
        if not await self._ensure_ready():
            return False

        try:
            document = self._document_text(title, content)

            def _upsert():
                embedding = self._model.encode(
                    [document],
                    normalize_embeddings=True,
                ).tolist()[0]
                self._collection.upsert(
                    ids=[note_id],
                    embeddings=[embedding],
                    documents=[document],
                    metadatas=[{
                        "telegram_id": str(telegram_id),
                        "title": title,
                    }],
                )

            await asyncio.to_thread(_upsert)
            return True
        except Exception:
            return False

    async def delete(self, note_id: str) -> bool:
        if not await self._ensure_ready():
            return False

        try:
            await asyncio.to_thread(self._collection.delete, ids=[note_id])
            return True
        except Exception:
            return False

    async def search(
        self,
        *,
        telegram_id: int,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not await self._ensure_ready():
            return []

        try:
            def _query():
                embedding = self._model.encode(
                    [query],
                    normalize_embeddings=True,
                ).tolist()[0]
                return self._collection.query(
                    query_embeddings=[embedding],
                    n_results=max(1, min(limit, 20)),
                    where={"telegram_id": str(telegram_id)},
                    include=["documents", "metadatas", "distances"],
                )

            result = await asyncio.to_thread(_query)
            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            items: list[dict[str, Any]] = []
            for index, note_id in enumerate(ids):
                metadata = metadatas[index] if index < len(metadatas) else {}
                items.append({
                    "id": str(note_id),
                    "title": str(metadata.get("title") or ""),
                    "content": str(
                        documents[index] if index < len(documents) else ""
                    ),
                    "semantic_distance": (
                        float(distances[index])
                        if index < len(distances) and distances[index] is not None
                        else None
                    ),
                })
            return items
        except Exception:
            return []
