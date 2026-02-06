from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncAzureOpenAI
from pydantic import BaseModel, ConfigDict, Field

from models.vector_db_model import SearchChunk
from models.llm_models import AnswerChunk, ChunkCitation, RankedChunks


class LlmService(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    async def rerank_chunks(
        self,
        *,
        query: str,
        chunks: list[SearchChunk],
        llm_model: str,
        split_every_k: int,
        timer: Any,
        max_concurrency: int = 4,
    ) -> list[SearchChunk]:
        raise NotImplementedError

    @abstractmethod
    async def answer_with_context(
        self,
        *,
        query: str,
        llm_model: str,
        chunk_hits: list[SearchChunk],
    ) -> tuple[str, list[ChunkCitation]]:
        raise NotImplementedError


class AzureOpenAILlmService(LlmService):
    async_client: AsyncAzureOpenAI = Field(exclude=True)

    async def rerank_chunks(
        self,
        *,
        query: str,
        chunks: list[SearchChunk],
        llm_model: str,
        split_every_k: int,
        timer: Any,
        max_concurrency: int = 4,
    ) -> list[SearchChunk]:
        import asyncio

        scored_chunks: list[tuple[int, str, SearchChunk]] = []
        assert all(c.uuid_str for c in chunks)
        uuid_to_chunk = {c.uuid_str: c for c in chunks}

        n_chunks = _split_by_size(chunks, chunk_size=split_every_k)
        sem = asyncio.Semaphore(max_concurrency)

        async def _rerank_batch(i: int, chs: list[SearchChunk]) -> None:
            contents_joined = " ".join(
                f"--- START #{idx}, Document uuid:{c.uuid_str or 'Unknown'} ---\n"
                f"{c.content or 'Unknown'}\n"
                f"--- END #{idx} Document {c.uuid_str or 'Unknown'} ---\n"
                for idx, c in enumerate(chs)
            )

            prompt = f"""
                        You are given {len(chs)} documents. For each document you MUST:
                        - Assign a relevance score on a scale from 0 to 10 (10 = highly relevant, 0 = irrelevant), determining how relevant this document is to the query

                        Query: {query}
                        Documents: {contents_joined}
                    """

            async with sem:
                with timer.start_timer(f"rerank_{i}"):
                    resp = await self.async_client.responses.parse(
                        model=llm_model,
                        input=[
                            {
                                "role": "system",
                                "content": "You're a helpful assistant. Your task is to evaluate the relevance of EACH document to the given query",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        text_format=RankedChunks,
                    )

            if resp.output_parsed and resp.output_parsed.ranked_chunks:
                ranked_cs = resp.output_parsed.ranked_chunks
                try:
                    for rc in ranked_cs:
                        scored_chunks.append((rc.score, rc.score_reason, uuid_to_chunk[rc.uuid_str]))
                except Exception as ex:
                    print(f"EX: {ex}\n{rc} \n{scored_chunks}\n{uuid_to_chunk}")
            else:
                print(f"Missing attrb in reranker {resp}")

        await asyncio.gather(*[_rerank_batch(i, chs) for i, chs in enumerate(n_chunks)])

        scored_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)
        return [tup[-1] for tup in scored_chunks]

    async def answer_with_context(
        self,
        *,
        query: str,
        llm_model: str,
        chunk_hits: list[SearchChunk],
    ) -> tuple[str, list[ChunkCitation]]:
        relevant_context = []
        relev_chunk_hits = chunk_hits
        assert all(c is not None for c in relev_chunk_hits)

        for chunk_h in relev_chunk_hits:
            chunk_format_str = (
                f"[ book: {chunk_h.book_name} ; chunk_nr: {chunk_h.chunk_id} ] || {chunk_h.content} ||"
            )
            relevant_context.append(chunk_format_str)

        system = (
            "You answer using ONLY the provided list of content chunks. If the content chunks aren't relevant to answer the query, you reply with 'I dont know based on the given context.'\n"
            "Each chunk has a header denoted by '[' and ']'. The content of the chunk is denoted by: '||'\n"
            "Include a brief 'Sources' list with chunk uuids and their book_name."
        )
        joined_context = ">>".join(relevant_context)
        joined_context[: joined_context.rfind(">> ")]
        prompt = f"""Question: {query}
                    Context:
                    {joined_context}
                    """

        llm_answer: AnswerChunk | str = "No matches found with query. Ensure that book index is populated."
        if len(relev_chunk_hits) == 0 and len(chunk_hits) > 0:
            llm_answer = "Matches found, but none were relevant."
            relev_chunk_hits = chunk_hits
        elif len(relev_chunk_hits) > 0:
            resp = await self.async_client.responses.parse(
                model=llm_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                text_format=AnswerChunk,
            )
            llm_answer = resp.output_parsed

        if isinstance(llm_answer, AnswerChunk):
            return llm_answer.answer, llm_answer.used_chunks

        return llm_answer, []


class MockLlmService(LlmService):
    answer: str = "Mock answer: integration test response."
    randomize: bool = False

    async def rerank_chunks(
        self,
        *,
        query: str,
        chunks: list[SearchChunk],
        llm_model: str,
        split_every_k: int,
        timer: Any,
        max_concurrency: int = 4,
    ) -> list[SearchChunk]:
        return list(chunks)

    async def answer_with_context(
        self,
        *,
        query: str,
        llm_model: str,
        chunk_hits: list[SearchChunk],
    ) -> tuple[str, list[ChunkCitation]]:
        answer = self.answer
        if self.randomize:
            answer = f"{self.answer} Seed:{random.randint(0, 9999)}"

        citations: list[ChunkCitation] = []
        if chunk_hits:
            top = chunk_hits[0]
            citations.append(
                ChunkCitation(
                    book_name=top.book_name,
                    chunk_content=top.content or "",
                    chunk_nr=top.chunk_id,
                )
            )

        return answer, citations


def _split_by_size(data: list, chunk_size: int) -> list[list]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
