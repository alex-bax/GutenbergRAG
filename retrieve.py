from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorQuery
from openai import AzureOpenAI

from azure.core.paging import ItemPaged

from preprocess_book import create_embeddings
from typing import Any


def _search_chunks(*, query: str, 
                  search_client:SearchClient, 
                  embed_client:AzureOpenAI, 
                  embed_model_deployed:str, k=5) -> list[dict[str, Any]]:
    
    query_emb_vec = create_embeddings(texts=[query], 
                              embed_client=embed_client, 
                              model_deployed=embed_model_deployed)[0]
    
    vec_q = VectorizedQuery(vector=query_emb_vec, k_nearest_neighbors=40, fields="contentVector")

    results:ItemPaged = search_client.search(
        search_text=query,                         # hybrid: BM25 + vector
        vector_queries=[vec_q],
        top=k,
        # query_type="semantic",      # TODO: use this one?
        # semantic_configuration_name="default"
    )
    
    hits=[]
    for r in results:
        hits.append({
            "score": r["@search.score"],
            "book": r["book"], "chapter": r["chapter"], "chunk_id": r["chunk_id"],
            "content": r["content"]
        })
    
    return hits


def answer(*, query: str, 
           search_client:SearchClient, 
           embed_client:AzureOpenAI, 
           llm_client:AzureOpenAI,
           embed_model_deployed="text-embedding-3-small",
           llm_model_deployed="gpt-5-mini") -> dict[str, Any]:
    
    hits = _search_chunks(query=query, 
                         search_client=search_client, 
                         embed_client=embed_client, 
                         embed_model_deployed=embed_model_deployed, 
                         k=6)
    context_blocks = []
    citations = []

    for h in hits:
        context_blocks.append(f"[{h['chapter']} · chunk {h['chunk_id']}] {h['content']}")
        citations.append({"chapter": h["chapter"], "chunk_id": h["chunk_id"]})

    system = """You answer using only the provided context. 
                If unsure, say you don't know. 
                Include a brief 'Sources' list with chapter + chunk ids.
                """
    prompt = f"""Question: {query}
                Context:
                {chr(10).join(context_blocks)}
                """

    chat = llm_client.responses.create(
        model=llm_model_deployed,
        input=[
            {"role":"system","content":system},
            {"role":"user","content":prompt}
        ]
    )
    return {"answer": chat.output_text, "citations": citations}