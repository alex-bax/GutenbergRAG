from openai import AzureOpenAI, AsyncAzureOpenAI
from settings import Settings, get_settings
from pyrate_limiter import Limiter
from constants import MIN_SEARCH_SCORE
from db.vector_store_abstract import AsyncVectorStore
from models.api_response_model import QueryResponse
from embedding_pipeline import create_embeddings_async
from models.vector_db_model import SearchChunk
from pydantic import Field, BaseModel

class RankedChunk(BaseModel):
    score:int = Field(ge=0, le=10)
    score_reason:str = Field(..., description="The reasoning for choosing the score")
    content:str

async def search_chunks(*, query: str, 
                        vector_store:AsyncVectorStore, 
                        embed_client:AsyncAzureOpenAI, 
                        llm_client:AzureOpenAI,
                        embed_model_deployed:str, 
                        tok_lim:Limiter,
                        req_lim:Limiter,
                        k=10
                        ) -> list[SearchChunk]: 
    
    query_emb_vec = await create_embeddings_async(inp_batches=[[query]], 
                                                embed_client=embed_client, 
                                                model_deployed=embed_model_deployed,
                                                tok_limiter=tok_lim,
                                                req_limiter=req_lim
                                                )

    results:list[SearchChunk] = await vector_store.search_by_embedding(
                                                embed_query_vector=query_emb_vec[0],
                                                filter=None,
                                                k=k
                                            )
    ranked_results = simple_llm_reranker(q=query, chunks=results, llm_client=llm_client)

    return ranked_results


def simple_llm_reranker(q:str, chunks:list[SearchChunk], llm_client:AzureOpenAI) -> list[SearchChunk]:
    scored_chunks:list[tuple[int, str, SearchChunk]] = []
    for c in chunks:
        prompt = f"""
                    Query: {q}
                    Document: {c.content}
                    
                    On a scale of 0-10, how relevant is this document to the query?
                    Provide your score and brief reasoning.
                """
        resp = llm_client.responses.parse(
            input=[
                {"role":"system","content":"You're a helpful assistant. Your task is to evaluate the relevance of the document to the given query"},
                {"role":"user", "content":prompt}
            ],
            text_format=RankedChunk
        )
        if resp.output_parsed:
            ranked_c = resp.output_parsed
            c.rank = ranked_c.score
            c.rank_reason = ranked_c.score_reason
            scored_chunks.append((ranked_c.score, ranked_c.score_reason, c))

    scored_chunks = sorted(scored_chunks, key=lambda x:x[0])

    return [tup[-1] for tup in scored_chunks]


# TODO: make async - use AzureAsync package
def answer_with_context(*, query:str, 
                        llm_client:AzureOpenAI, 
                        llm_model_deployed:str, 
                        chunk_hits:list[SearchChunk]) -> tuple[str, list[SearchChunk]]:
   
    relevant_context = []

    relev_chunk_hits = [c for c in chunk_hits if c.search_score >= MIN_SEARCH_SCORE]
    assert all(c is not None for c in relev_chunk_hits)

    for chunk_h in relev_chunk_hits:
        chunk_format_str = f"[ book {chunk_h.book_name} ; chunk_id {chunk_h.uuid_str} ; search score {chunk_h.search_score}] || {chunk_h.content} ||"
        relevant_context.append(chunk_format_str)

    ## TODO! work on this. the delims for contetn
    system = """You answer using only the provided list of content chunks. 
                Each chunk has a header denoted by '[' and ']'. The content of the chunk is denoted by: '|'
                
                If unsure, say you don't know. 
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
        chat = llm_client.responses.create(
            model=llm_model_deployed,
            input=[
                {"role":"system","content":system},
                {"role":"user","content":prompt}
            ]
        )
        llm_answer = chat.output_text

    return llm_answer, relev_chunk_hits


async def answer_rag(*, query: str, 
                    sett:Settings,
                    top_n_matches:int,
                    ) -> QueryResponse:
                
    req_lim, tok_lim = sett.get_limiters()

    ranked_chunks = await search_chunks(query=query, 
                                    vector_store=await sett.get_vector_store(), 
                                    embed_client=sett.get_async_emb_client(), 
                                    embed_model_deployed=sett.EMBED_MODEL_DEPLOYMENT, 
                                    tok_lim=tok_lim,
                                    req_lim=req_lim,
                                    k=top_n_matches,
                                    llm_client=sett.get_llm_client())

    top_chunks = ranked_chunks[:4]


    llm_answer, relevant_chunks = answer_with_context(query=query, 
                                                      llm_client=sett.get_llm_client(), 
                                                      llm_model_deployed=sett.AZ_OPENAI_MODEL_DEPLOYMENT, 
                                                      chunk_hits=top_chunks)    
    
    return QueryResponse(answer=llm_answer, citations= relevant_chunks)



async def run_gutenberg_rag(question: str, sett:Settings) -> tuple[str, list[str]]:
    """
    Entire RAG retrival hook
    Returns:
        - answer: str                   (model's final answer)
        - contexts: list[str]           (list of retrieved passages / chunks)
    """
    
    q_resp = await answer_rag(query=question, sett=sett, top_n_matches=10)
    
    contexts_found = [c.content for c in q_resp.citations if c.content]
    return q_resp.answer, contexts_found