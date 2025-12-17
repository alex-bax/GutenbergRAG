from __future__ import annotations
import asyncio
from typing import Any

from pydantic import Field, ConfigDict
from llama_index.core.embeddings import BaseEmbedding

from embedding_pipeline import batch_texts_by_tokens, create_embeddings_async
from models.vector_db_model import EmbeddingVec

Vector = list[float]


def _run_async_only_if_no_loop(coro):
    """Run async coro only when no loop is running in this thread."""
    try:
        asyncio.get_running_loop()
        # If we get here, a loop is running -> cannot block it safely.
        raise RuntimeError(
            "A running event loop was detected. "
            "This code path attempted to call async embeddings from a sync method. "
            "Ensure you are using the async embedding methods (_aget_*) and not the sync ones."
        )
    except RuntimeError as e:
        # Two different RuntimeErrors can occur:
        # 1) get_running_loop() raises RuntimeError when *no loop* exists (good)
        # 2) we raised our own RuntimeError above (bad)
        if "no running event loop" in str(e).lower():
            return asyncio.run(coro)
        raise


def _to_vector(v: Any) -> Vector:
    
    if isinstance(v, list):
        return [float(x) for x in v]
    elif isinstance(v, EmbeddingVec):
        return v.vector
    
    try:
        return [float(x) for x in v]
    except TypeError as e:
        raise TypeError(f"Cannot convert embedding of type {type(v)} to list[float]") from e


class RateLimitedAzureEmbedding(BaseEmbedding):
    embed_client: Any = Field(exclude=True)
    deployment_name: str

    tok_limiter: Any = Field(exclude=True)
    req_limiter: Any = Field(exclude=True)

    batch_size: int = Field(..., description="Size of each text batch. Not to be confused with chunk size. Used as limit to stay within the max token size limit of the embedding model")
    # _embed_dim: Optional[int] = Field(default=None, repr=False)
    embed_dim_value: int|None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def embed_dim(self) -> int|None:
        return self.embed_dim_value

    #  SYNC methods (required abstract methods) 
    def _get_text_embedding(self, text: str) -> Vector:
        return _to_vector(_run_async_only_if_no_loop(self._aget_text_embedding(text)))

    def _get_query_embedding(self, query: str) -> Vector:
        return _to_vector(_run_async_only_if_no_loop(self._aget_query_embedding(query)))

    def _get_text_embeddings(self, texts: list[str]) -> list[Vector]:
        vecs = _run_async_only_if_no_loop(self._aget_text_embeddings(texts))
        return [_to_vector(v) for v in vecs]

    #  ASYNC methods 
    async def _aget_query_embedding(self, query: str) -> Vector:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> Vector:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Vector]:
        inp_batches = batch_texts_by_tokens(
                                texts=texts,
                                max_tokens_per_request=self.batch_size,
                            )

        vectors = await create_embeddings_async(
                            embed_client=self.embed_client,
                            model_deployed=self.deployment_name,
                            inp_batches=inp_batches,
                            tok_limiter=self.tok_limiter,
                            req_limiter=self.req_limiter,
                        )

        # Forcing list[list[float]] for LlamaIndex
        return [_to_vector(v) for v in vectors]
