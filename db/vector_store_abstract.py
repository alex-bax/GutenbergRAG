from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel 
from typing import Any, Sequence
from models.vector_db_model import ContentUploadChunk


class VectorStore(ABC, BaseModel):
    """Backend-agnostic vector db interface."""

    @abstractmethod
    def upsert(*, self, chunks: list[ContentUploadChunk]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns a list of hits (each hit is a dict with at least: id, score, payload).
        """
        ...

    @abstractmethod
    def delete(*, self, ids: list[str]) -> None:
        ...

    @abstractmethod
    def create_index(*, self, ids: list[str]) -> None:
        ...
