from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel 
from typing import Sequence
from models.vector_db_model import UploadChunk, SearchChunk, EmbeddingVec

# TODO: add paginated_search

class AsyncVectorStore(BaseModel, ABC):
    """Backend-agnostic vector db interface."""

    @abstractmethod
    async def upsert(self, *, chunks: Sequence[UploadChunk]) -> None:
        ...

    @abstractmethod
    async def get_missing_ids(self, *, book_ids:set[int]) -> set[int]:
        """Return a list of book_ids from NOT present from given list"""
        ...

    @abstractmethod
    async def search_by_embedding(self, *,
                                    embed_query_vector:EmbeddingVec,
                                    k: int = 10,
                                ) -> list[SearchChunk]:
        """
        Returns a list of hits (each hit is a dict with at least: id, score, payload).
        """
        ...


    @abstractmethod
    async def delete_books(self, *, book_ids:Sequence[int]) -> None:
        ...

    @abstractmethod
    async def create_missing_collection(self, *, collection_name:str) -> None:
        ...
