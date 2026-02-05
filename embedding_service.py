from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import hashlib
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, ConfigDict, BaseModel

from config.params import EmbeddingDimension
from embedding_pipeline import batch_texts_by_tokens, create_embeddings_async
from models.vector_db_model import EmbeddingVec

Vector = list[float]


class EmbeddingService(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[EmbeddingVec]:
        """Return embeddings for each input text."""
        raise NotImplementedError


class AzureOpenAIEmbeddingService(EmbeddingService):
    embed_client: Any = Field(exclude=True)
    deployment_name: str = "text-embedding-3-small"
    tok_limiter: Any = Field(exclude=True)
    req_limiter: Any = Field(exclude=True)
    batch_size: int

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingVec]:
        inp_batches = batch_texts_by_tokens(
            texts=texts,
            max_tokens_per_request=self.batch_size,
        )
        return await create_embeddings_async(
            embed_client=self.embed_client,
            model_deployed=self.deployment_name,
            inp_batches=inp_batches,
            tok_limiter=self.tok_limiter,
            req_limiter=self.req_limiter,
        )


class MockEmbeddingService(EmbeddingService):
    dim: EmbeddingDimension = EmbeddingDimension.SMALL
    salt: str = "mock"

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingVec]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> EmbeddingVec:
        digest = hashlib.sha256(f"{self.salt}:{text}".encode("utf-8")).digest()
        base = [b / 255.0 for b in digest]
        dim_value = int(self.dim)
        vector = [base[i % len(base)] for i in range(dim_value)]
        return EmbeddingVec(vector=vector, dim=self.dim)


def _run_async_only_if_no_loop(coro):
    """Run async coro only when no loop is running in this thread."""
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "A running event loop was detected. "
            "This code path attempted to call async embeddings from a sync method. "
            "Ensure you are using the async embedding methods (_aget_*) and not the sync ones."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            return asyncio.run(coro)
        raise


def _to_vector(v: Any) -> Vector:
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, EmbeddingVec):
        return v.vector
    try:
        return [float(x) for x in v]
    except TypeError as e:
        raise TypeError(f"Cannot convert embedding of type {type(v)} to list[float]") from e


class EmbeddingServiceLlamaIndexAdapter(BaseEmbedding):
    embedding_service: EmbeddingService = Field(exclude=True)
    embed_dim_value: int | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def embed_dim(self) -> int | None:
        return self.embed_dim_value

    def _get_text_embedding(self, text: str) -> Vector:
        return _to_vector(_run_async_only_if_no_loop(self._aget_text_embedding(text)))

    def _get_query_embedding(self, query: str) -> Vector:
        return _to_vector(_run_async_only_if_no_loop(self._aget_query_embedding(query)))

    def _get_text_embeddings(self, texts: list[str]) -> list[Vector]:
        vecs = _run_async_only_if_no_loop(self._aget_text_embeddings(texts))
        return [_to_vector(v) for v in vecs]

    async def _aget_query_embedding(self, query: str) -> Vector:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> Vector:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Vector]:
        vectors = await self.embedding_service.embed_texts(texts)
        return [_to_vector(v) for v in vectors]
