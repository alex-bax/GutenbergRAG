from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
# from azure.search.documents.models import VectorizedQuery, VectorQuery
from openai import AzureOpenAI

from constants import MIN_SEARCH_SCORE

from db.vector_store_abstract import AsyncVectorStore
from models.api_response_model import QueryResponse
from ingestion.preprocess_book import create_embeddings
from models.vector_db_model import SearchChunk

async def search_chunks(*, query: str, 
                        search_client:AsyncVectorStore, 
                        embed_client:AzureOpenAI, 
                        embed_model_deployed:str, k=5) -> list[SearchChunk]: #list[dict[str, Any]]:
    
    query_emb_vec = create_embeddings(batches=[query], 
                                        embed_client=embed_client, 
                                        model_deployed=embed_model_deployed)[0]

    results:list[SearchChunk] = await search_client.search_by_embedding(
                                                embed_query_vector=query_emb_vec,
                                                k=10
                                            )
    return results


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



async def answer_api(*, query: str, 
           search_client:AsyncVectorStore, 
           embed_client:AzureOpenAI, 
           llm_client:AzureOpenAI,
           top_n_matches:int,
           embed_model_deployed:str,
           llm_model_deployed:str) -> QueryResponse:
    
    chunk_hits = await search_chunks(query=query, 
                         search_client=search_client, 
                         embed_client=embed_client, 
                         embed_model_deployed=embed_model_deployed, 
                         k=top_n_matches)
   

    llm_answer, relevant_chunks = answer_with_context(query=query, 
                                                      llm_client=llm_client, 
                                                      llm_model_deployed=llm_model_deployed, 
                                                      chunk_hits=chunk_hits)    
    
    return QueryResponse(answer=llm_answer, citations= relevant_chunks)