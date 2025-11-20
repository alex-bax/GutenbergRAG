from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel 
from typing import Any
from models.vector_db_model import ContentUploadChunk, SearchChunk, EmbeddingVec

# TODO: add paginated_search

class AsyncVectorStore(ABC, BaseModel):
    """Backend-agnostic vector db interface."""

    @abstractmethod
    async def upsert(*, self, chunks: list[ContentUploadChunk]) -> None:
        ...

    @abstractmethod
    async def get_missing_ids(self, book_ids:list[int]) -> list[int]:
        ...

    @abstractmethod
    async def search_chunks(
        self,
        embed_query_vector:EmbeddingVec,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchChunk]:
        """
        Returns a list of hits (each hit is a dict with at least: id, score, payload).
        """
        ...

    @abstractmethod
    async def delete(*, self, ids: list[str]) -> None:
        ...

    @abstractmethod
    async def create_index(*, self, ids: list[str]) -> None:
        ...
