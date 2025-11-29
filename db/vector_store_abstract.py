from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel 
from typing import Sequence, Any
from models.vector_db_model import UploadChunk, SearchChunk, SearchPage, EmbeddingVec
from models.api_response_model import GBBookMeta
# TODO: add paginated_search

class AsyncVectorStore(BaseModel, ABC):
    """Backend-agnostic vector db interface."""

    @abstractmethod
    async def upsert_chunks(self, *, chunks: Sequence[UploadChunk]) -> None:
        ...


    @abstractmethod
    async def delete_books(self, *, book_ids:set[int]) -> None:
        ...


    @abstractmethod
    async def get_missing_ids_in_store(self, *, book_ids:set[int]) -> set[int]:
        """Return a list of book_ids NOT present from given list"""
        ...


    @abstractmethod
    async def search_by_embedding(self, *,
                                    embed_query_vector:EmbeddingVec,
                                    filter:dict[str,Any]|None=None,
                                    k: int = 10,
                                ) -> list[SearchChunk]:
        """
        Returns a list of chunks.
        """
        ...

    
    @abstractmethod
    async def get_paginated_chunks_by_book_ids(self, *, book_ids:set[int]) -> SearchPage:
        ...
    
    @abstractmethod
    async def get_chunk_by_nr(self, *, chunk_nr:int, book_id:int) -> SearchPage:
        ...
    
    @abstractmethod
    async def get_chunk_count_in_book(self, *, book_id:int) -> int:
        ...


    @abstractmethod
    async def paginated_search_by_text(self, *, text_query:str, limit:int, skip:int, cont_token:str|None=None) -> SearchPage:
        """Return a list of chunks matching with the text_query argument"""
        ...


    @abstractmethod
    async def create_missing_collection(self, *, collection_name:str) -> None:
        ...

    @abstractmethod
    async def delete_collection(self, *, collection_name:str) -> None:
        ...


    # @abstractmethod
    # async def populate_small_collection(self) -> list[GBBookMeta]:
    #     """
    #     Populate the vector collection with a small default list of books, if they're missing: 
    #         1661,The Adventures of Sherlock Holmes,
    #         84,Frankenstein,Mary Shelley,
    #         2701,Moby-Dick,Herman Melville

    #     Returns list of books actually uploaded.
    #     """
    #     ...
