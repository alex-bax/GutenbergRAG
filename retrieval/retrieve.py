from evals.timer_helper import Timer
from openai import AzureOpenAI, AsyncAzureOpenAI
from config.settings import Settings
from embedding_service import EmbeddingService
# from config.hyperparams import MIN_SEARCH_SCORE
from db.vector_store_abstract import AsyncVectorStore
from llm_service import LlmService
from models.api_response_model import QueryResponse
from models.llm_models import ChunkCitation
from models.vector_db_model import SearchChunk
from pydantic import Field, BaseModel
from vector_store_utils import split_by_size
from monitor_metrics.rag_metrics import rag_stage_seconds


async def search_chunks(*, query: str, 
                        vector_store:AsyncVectorStore, 
                        embedding_service: EmbeddingService,
                        keep_top_k:int,
                        ) -> list[SearchChunk]: 
    print(f'TOP K : {keep_top_k}')

    query_emb_vec = await embedding_service.embed_texts([query])

    results:list[SearchChunk] = await vector_store.search_by_embedding(
                                                embed_query_vector=query_emb_vec[0],
                                                filter=None,
                                                k=keep_top_k
                                            )
    return results


async def async_llm_reranker(
    *,
    q: str,
    chunks: list[SearchChunk],
    llm_service: LlmService,
    llm_model: str,
    split_every_k: int,
    timer: Timer,
    max_concurrency: int = 4,
) -> list[SearchChunk]:
    with timer.start_timer("rerank_total"):
        return await llm_service.rerank_chunks(
            query=q,
            chunks=chunks,
            llm_model=llm_model,
            split_every_k=split_every_k,
            timer=timer,
            max_concurrency=max_concurrency,
        )


async def answer_with_context(
    *,
    query: str,
    llm_service: LlmService,
    llm_model_deployed: str,
    chunk_hits: list[SearchChunk],
) -> tuple[str, list[ChunkCitation]]:
    return await llm_service.answer_with_context(
        query=query,
        llm_model=llm_model_deployed,
        chunk_hits=chunk_hits,
    )


async def answer_rag(*, query: str, 
                    sett:Settings,
                    keep_top_k:int,
                    timer:Timer
                    ) -> QueryResponse:
        
    hp = sett.get_hyperparams()
    embedding_service = sett.get_embedding_service()
    llm_service = sett.get_llm_service()

    with timer.start_timer("search"):
        unranked_chunks = await search_chunks(query=query, 
                                            vector_store=await sett.get_vector_store(), 
                                            embedding_service=embedding_service,
                                            keep_top_k=keep_top_k,
                                        )
    rag_stage_seconds.labels(stage="search").observe(timer.timings["search"])

    ranked_chunks = await async_llm_reranker(
        q=query,
        chunks=unranked_chunks,
        llm_service=llm_service,
        llm_model=hp.rerank.model,
        split_every_k=hp.rerank.batch_size,
        timer=timer,
    )
    
    top_chunks = ranked_chunks[:hp.generation.num_context_chunks]      
    rag_stage_seconds.labels(stage="rerank_total").observe(timer.timings["rerank_total"])
    

    with timer.start_timer("answer_with_contexts"):
        llm_answer, relevant_chunks = await answer_with_context(
            query=query,
            llm_service=llm_service,
            llm_model_deployed=sett.AZ_OPENAI_MODEL_DEPLOYMENT,
            chunk_hits=top_chunks,
        )
    rag_stage_seconds.labels(stage="answer_with_contexts").observe(timer.timings["answer_with_contexts"])

    return QueryResponse(answer=llm_answer, citations=top_chunks)



async def run_gutenberg_rag(question: str, sett:Settings, timer:Timer) -> tuple[str, list[str]]:
    """
    Entire RAG retrival hook
    Returns:
        - answer: str                   (model's final answer)
        - contexts: list[str]           (list of retrieved passages / chunks)
    """
    
    q_resp = await answer_rag(query=question, 
                              sett=sett, 
                              keep_top_k=sett.get_hyperparams().retrieval.top_k,#15, 
                              timer=timer)
    
    contexts_found = [c.content for c in q_resp.citations if c.content]
    return q_resp.answer, contexts_found
