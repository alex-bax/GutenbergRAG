from __future__ import annotations

from typing import Any, Sequence
from pydantic import Field
from pydantic_settings import SettingsConfigDict  # if you want config
from .vector_store_abstract import AsyncVectorStore
from models.api_response_model import SearchChunk, SearchPage
from models.vector_db_model import UploadChunk, EmbeddingVec
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import numpy as np

class InMemoryVectorStore(AsyncVectorStore):
    """
    Simple in-memory implementation of AsyncVectorStore for tests.

    Stores chunks in a dict keyed by book_id. Good enough for:
      - upload_missing_book_ids
      - routes that index books
      - delete/reset between tests
    """

    # pydantic v2 config (optional but often useful)
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    # internal state
    # book_id -> list of chunks
    data: dict[int, list[UploadChunk]] = Field(default_factory=dict)
    
    collections: set[str] = Field(default_factory=set)

    async def create_missing_collection(self, *, collection_name: str) -> None:
        self.collections.add(collection_name)


    async def delete_collection(self, *, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.data.clear()


    async def upsert_chunks(self, *, chunks: Sequence[UploadChunk]) -> None:
        # Very naive: group by book id and append
        for chunk in chunks:
            book_id = chunk.book_id  
            self.data.setdefault(book_id, []).append(chunk)


    async def delete_books(self, *, book_ids: set[int]) -> None:
        for book_id in book_ids:
            self.data.pop(book_id, None)


    async def get_missing_ids_in_store(self, *, book_ids: set[int]) -> set[int]:
        existing_ids = set(self.data.keys())
        return book_ids - existing_ids


    async def search_by_embedding(
        self,
        *,
        embed_query_vector: EmbeddingVec,
        filter: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[SearchChunk]:
        candidates:list[tuple[float, UploadChunk]] = []

        for book_id, chunks in self.data.items():
            for chunk in chunks:
                chunk_emb = chunk.content_vector 
                score = distance.cosine(np.array(embed_query_vector.vector), 
                                        np.array(chunk_emb.vector))
                candidates.append((score, chunk))

        # sort by score descending, take top k
        candidates.sort(key=lambda t: t[0], reverse=True)
        top = candidates[:k]

        results: list[SearchChunk] = []
        for score, chunk in top:
            result = SearchChunk(**chunk.model_dump(), search_score=score)
            results.append(result)

        return results
    

    async def get_paginated_chunks_by_book_ids(
        self,
        *,
        book_ids: set[int],
    ) -> SearchPage:
        raise NotImplementedError("Not used in current tests. Implement when needed.")


    async def get_chunk_by_nr(self, *, chunk_nr: int, book_id: int) -> SearchPage:
        raise NotImplementedError("Not used in current tests. Implement when needed.")


    async def get_chunk_count_in_book(self, *, book_id: int) -> int:
        # Could return len(...) if you want, or stub until needed
        chunks = self.data.get(book_id, [])
        return len(chunks)


    async def paginated_search_by_text(
        self,
        *,
        text_query: str,
        limit: int,
        skip: int,
        continuation_token: str | None = None,
    ) -> SearchPage:
        raise NotImplementedError("Not used in current tests. Implement when needed.")
