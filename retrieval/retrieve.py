from evals.timer_helper import Timer
from openai import AzureOpenAI, AsyncAzureOpenAI
from config.settings import Settings, get_settings
from pyrate_limiter import Limiter
# from config.hyperparams import MIN_SEARCH_SCORE
from db.vector_store_abstract import AsyncVectorStore
from models.api_response_model import QueryResponse
from embedding_pipeline import create_embeddings_async
from models.vector_db_model import SearchChunk
from pydantic import Field, BaseModel
from vector_store_utils import _split_by_size
from metrics.rag_metrics import rag_stage_seconds

class RankedChunk(BaseModel):
    score:int = Field(ge=0, le=10)
    score_reason:str = Field(..., description="The reasoning for choosing the score")
    # content:str
    uuid_str:str

class RankedChunks(BaseModel):
    ranked_chunks:list[RankedChunk]

class ChunkCitation(BaseModel):
    book_name: str
    chunk_content:str
    chunk_nr:int

class AnswerChunk(BaseModel):
    answer:str
    used_chunks:list[ChunkCitation]


async def search_chunks(*, query: str, 
                        vector_store:AsyncVectorStore, 
                        embed_client:AsyncAzureOpenAI, 
                        embed_model_deployed:str, 
                        tok_lim:Limiter,
                        req_lim:Limiter,
                        keep_top_k:int,
                        ) -> list[SearchChunk]: 
    print(f'TOP K : {keep_top_k}')

    query_emb_vec = await create_embeddings_async(inp_batches=[[query]], 
                                                embed_client=embed_client, 
                                                model_deployed=embed_model_deployed,
                                                tok_limiter=tok_lim,
                                                req_limiter=req_lim
                                                )

    results:list[SearchChunk] = await vector_store.search_by_embedding(
                                                embed_query_vector=query_emb_vec[0],
                                                filter=None,
                                                k=keep_top_k
                                            )
    return results


def simple_llm_reranker(q:str, chunks:list[SearchChunk], 
                        llm_client:AzureOpenAI, 
                        llm_model:str,
                        split_every_k:int,
                        timer:Timer
                        ) -> list[SearchChunk]:
    
    with timer.start_timer("rerank_total"):
        scored_chunks:list[tuple[int, str, SearchChunk]] = []
        assert all(c.uuid_str for c in chunks)
        uuid_to_chunk = {c.uuid_str:c for c in chunks}

        n_chunks = _split_by_size(chunks, chunk_size=split_every_k)
        for i, chs in enumerate(n_chunks):
            contents_joined = " ".join([f"--- START #{i}, Document uuid:{c.uuid_str} ---\n"+c.content+f"\n--- END #{i} Document {c.uuid_str}---\n" for i,c in enumerate(chs)])
            # print(contents_joined)
            prompt = f"""
                        You are given {len(chs)} documents. For each document you MUST:
                        - Assign a relevance score on a scale from 0 to 10 (10 = highly relevant, 0 = irrelevant), determining how relevant this document is to the query

                        Query: {q}
                        Documents: {contents_joined}
                    """

            with timer.start_timer(f"rerank_{i}"):
                resp = llm_client.responses.parse(      
                            model=llm_model,
                            input=[
                                {"role":"system","content":"You're a helpful assistant. Your task is to evaluate the relevance of EACH document to the given query"},
                                {"role":"user", "content":prompt}
                            ],
                            text_format=RankedChunks
                        )
            
            if resp.output_parsed and resp.output_parsed.ranked_chunks:
                ranked_cs = resp.output_parsed.ranked_chunks
                try:
                    for rc in ranked_cs:
                        scored_chunks.append((rc.score, rc.score_reason, uuid_to_chunk[rc.uuid_str]))
                except Exception as ex:
                    print(f'EX: {ex}\n{rc} \n{scored_chunks}\n{uuid_to_chunk}')
            else:
                print(f"Missing attrb in reranker {resp}")

    scored_chunks = sorted(scored_chunks, key=lambda x:x[0], reverse=True)
    
    return [tup[-1] for tup in scored_chunks]


# TODO: make async - use AzureAsync package
def answer_with_context(*, query:str, 
                        llm_client:AzureOpenAI, 
                        llm_model_deployed:str,
                        chunk_hits:list[SearchChunk]) -> tuple[str, list[SearchChunk]]:
   
    relevant_context = []

    relev_chunk_hits = chunk_hits #[c for c in chunk_hits if c.search_score >= MIN_SEARCH_SCORE]        # TODO: remove?
    assert all(c is not None for c in relev_chunk_hits)

    for chunk_h in relev_chunk_hits:
        chunk_format_str = f"[ book: {chunk_h.book_name} ; chunk_nr: {chunk_h.chunk_id} ] || {chunk_h.content} ||"
        relevant_context.append(chunk_format_str)

    ## TODO! work on this. the delims for contetn
    system = """You answer using ONLY the provided list of content chunks. If the content chunks aren't relevant to answer the query, you reply with 'I dont know based on the given context.'
                Each chunk has a header denoted by '[' and ']'. The content of the chunk is denoted by: '||'
                Include a brief 'Sources' list with chunk uuids and their book_name.
            """
    joined_context = ">>".join(relevant_context)
    joined_context[:joined_context.rfind(">> ")]
    prompt = f"""Question: {query}
                Context:
                {joined_context}      
                """

    llm_answer = "No matches found with query. Ensure that book index is populated."
    if len(relev_chunk_hits) == 0 and len(chunk_hits) > 0:
        llm_answer = "Matches found, but none were relevant."
        relev_chunk_hits = chunk_hits   

    elif len(relev_chunk_hits) > 0:
        # chat = llm_client.responses.create(
        resp = llm_client.responses.parse(
            model=llm_model_deployed,
            input=[
                # TODO: add role?
                {"role":"system","content":system},
                {"role":"user","content":prompt}
            ],
            text_format=AnswerChunk
        )
        llm_answer = resp.output_parsed

    return llm_answer.answer, llm_answer.used_chunks    # type:ignore


async def answer_rag(*, query: str, 
                    sett:Settings,
                    keep_top_k:int,
                    timer:Timer
                    ) -> QueryResponse:
        
    req_lim, tok_lim = sett.get_limiters()
    hp = sett.get_hyperparams()

    with timer.start_timer("search"):
        unranked_chunks = await search_chunks(query=query, 
                                            vector_store=await sett.get_vector_store(), 
                                            embed_client=sett.get_async_emb_client(), 
                                            embed_model_deployed=sett.EMBED_MODEL_DEPLOYMENT, 
                                            tok_lim=tok_lim,
                                            req_lim=req_lim,
                                            keep_top_k=keep_top_k,
                                        )
    rag_stage_seconds.labels(stage="search").observe(timer.timings["search"])

    ranked_chunks = simple_llm_reranker(q=query, 
                                        chunks=unranked_chunks, 
                                        llm_client=sett.get_llm_client(), 
                                        llm_model=hp.rerank.model,
                                        split_every_k=hp.rerank.batch_size,
                                        timer=timer)
    
    top_chunks = ranked_chunks[:hp.generation.num_context_chunks]      
    rag_stage_seconds.labels(stage="rerank_total").observe(timer.timings["rerank_total"])
    

    with timer.start_timer("answer_with_contexts"):
        llm_answer, relevant_chunks = answer_with_context(query=query, 
                                                        llm_client=sett.get_llm_client(), 
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