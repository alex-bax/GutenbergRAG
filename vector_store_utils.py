from datetime import datetime
import uuid, tiktoken
from statistics import mean, median, stdev
import asyncio
# from azure.search.documents import SearchClient,SearchItemPaged

from pyrate_limiter import Limiter
from openai import AsyncAzureOpenAI
from pathlib import Path
from create_visualizations import plot_token_counts_bar
from ingestion.preprocess_book import clean_headers 
from ingestion.chunking import fixed_size_chunking
from config.settings import Settings, get_settings
from embedding_pipeline import _count_tokens, batch_texts_by_tokens, create_embeddings_async
from models.api_response_model import GBBookMeta
from models.vector_db_model import EmbeddingVec, UploadChunk
from db.vector_store_abstract import AsyncVectorStore
from config.params import EmbeddingDimension
from models.schema import DBBookChunkStats
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from rate_limited_llama_embedder import RateLimitedAzureEmbedding

def _split_by_size(data: list, chunk_size: int) -> list[list]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# OLD fixed size chunking
# async def upload_to_index_async(*, vec_store:AsyncVectorStore, 
#                                 embed_client:AsyncAzureOpenAI, 
#                                 token_limiter:Limiter,
#                                 request_limiter:Limiter,
#                                 raw_book_content: str,
#                                 book_meta: GBBookMeta,
#                             ) -> list[UploadChunk]:
#     sett = get_settings()

#     book_str = clean_headers(raw_book=raw_book_content) 
#     if len(book_str) == 0:      
#         print(f'** INFO No book content str extracted --- skipping {book_meta.title}')
#         return []

#     docs:list[UploadChunk] = []
#     vector_items_added = []

#     hp = sett.get_hyperparams().ingestion
#     chunks = fixed_size_chunking(text=book_str, chunk_size=hp.chunk_size)
#     batches = batch_texts_by_tokens(texts=chunks, max_tokens_per_request=hp.max_tokens_pr_req)

#     embeddings = await create_embeddings_async(embed_client=embed_client, 
#                                             model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
#                                             inp_batches=batches,
#                                             tok_limiter=token_limiter,
#                                             req_limiter=request_limiter
#                                             )
        
#     assert len(chunks) == len(embeddings)

#     for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
#         chapter_item = UploadChunk(
#                             uuid_str= str(uuid.uuid4()),
#                             book_name= book_meta.title,
#                             book_id = book_meta.id,
#                             chunk_nr= i,
#                             content= chunk,
#                             content_vector= emb_vec
#                         )
        
#         docs.append(chapter_item)#.to_dict())
#         vector_items_added.append(chapter_item)

#     docs_splitted = _split_by_size(data=docs, chunk_size=hp.chunk_size)
#     for doc_chunks in docs_splitted:
#         await vec_store.upsert_chunks(chunks=doc_chunks)

#     return vector_items_added


def calc_book_chunk_stats(all_chunks:list[UploadChunk], conf_id:int) -> DBBookChunkStats:
    token_counts = [c.token_count for c in all_chunks]
    book_name, book_id = all_chunks[0].book_name, all_chunks[0].book_id
    assert all(book_name == c.book_name for c in all_chunks), "calc_book_chunk_stats -> All book titles not equal" 
    assert all(book_id == c.book_id for c in all_chunks), "calc_book_chunk_stats -> All book ids not equal"

    if len(token_counts) == 1:
        token_std = 0.0
    else:
        token_std = stdev(token_counts)  # sample std dev

    return DBBookChunkStats(
        id=all_chunks[0].book_id,
        config_id_used=conf_id,
        title=all_chunks[0].book_name,
        char_count=sum([len(c.content) for c in all_chunks]),
        chunk_count=len(all_chunks),
        token_mean=mean(token_counts),
        token_median=median(token_counts),
        token_min=min(token_counts),
        token_max=max(token_counts),
        token_std=token_std,
        token_counts=token_counts
    )


async def async_upload_book_to_index(*, vec_store:AsyncVectorStore, 
                                embed_client:AsyncAzureOpenAI, 
                                token_limiter:Limiter,
                                request_limiter:Limiter,
                                raw_book_content: str,
                                book_meta: GBBookMeta,
                                sett:Settings,
                                time_started:str,
                                calc_chunk_stats=True
                            ) -> tuple[list[UploadChunk], DBBookChunkStats|None]:
    
    book_str = clean_headers(raw_book=raw_book_content) if not sett.is_test else raw_book_content
    
    if len(book_str) == 0:      
        return ([], None)

    upload_chunks:list[UploadChunk] = []
    vector_items_added:list[UploadChunk] = []
    
    hp = sett.get_hyperparams()

    embed_model = RateLimitedAzureEmbedding(
                        embed_client=embed_client,
                        deployment_name="text-embedding-3-small",
                        tok_limiter=token_limiter,
                        req_limiter=request_limiter,
                        batch_size=hp.ingestion.max_tokens_pr_req,
                        embed_dim_value=hp.ingestion.embed_dim,
                    )

    splitter = SemanticSplitterNodeParser(
                    embed_model=embed_model,
                    buffer_size=hp.ingestion.sem_split_buffer_size,
                    breakpoint_percentile_threshold=hp.ingestion.sem_split_break_percentile,
                )

    doc = Document(text=book_str, metadata=book_meta.model_dump())
    
    nodes = await asyncio.to_thread(splitter.get_nodes_from_documents, [doc])

    chunks: list[str] = [n.get_content() for n in nodes]
    embeddings = await embed_model._aget_text_embeddings(chunks)
    
    for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
        chapter_item = UploadChunk(
                            uuid_str=str(uuid.uuid4()),
                            book_name=book_meta.title,
                            book_id=book_meta.id,
                            chunk_id=i,
                            content=chunk,
                            content_vector=EmbeddingVec(vector=emb_vec, dim=EmbeddingDimension.SMALL),
                            char_count=len(chunk),
                            token_count=_count_tokens(chunk, enc=tiktoken.get_encoding("cl100k_base"))
                        )
        upload_chunks.append(chapter_item)
 
    await vec_store.upsert_chunks(chunks=upload_chunks)
    hp = sett.get_hyperparams()
    book_chunk_stats = calc_book_chunk_stats(all_chunks=upload_chunks, conf_id=hp.config_id)

    if calc_chunk_stats and not sett.is_test:
        token_count_parent_p = Path("imgs", "index_stats", time_started)
        token_count_parent_p.mkdir(parents=True, exist_ok=True)
        plot_token_counts_bar(token_counts=book_chunk_stats.token_counts,
                            save_folder_name=token_count_parent_p,
                            title=f"{book_chunk_stats.title}\nconf id:{hp.config_id}")

    return vector_items_added, book_chunk_stats



async def _try():
    from ingestion.book_loader import _load_gb_meta_local

    # skip default vec population with is_test=True
    sett = get_settings(hyperparam_p=Path("config", "hp-sem-ch.json"))
    req_lim, token_lim = sett.get_limiters()
    now = datetime.now().strftime("%d-%m-%Y_%H%M")
   
    p = Path("evals", "books", "alices-adventures-in-wonderland_carroll-lewis_11.json")
    cached_gb_meta = _load_gb_meta_local(path=p)
    with open(cached_gb_meta.path_to_content, "r", encoding="utf-8") as f:
        book_content = f.read()
    
    s = await async_upload_book_to_index(
                    vec_store=await sett.get_vector_store(),
                    embed_client=sett.get_async_emb_client(),
                    token_limiter=token_lim,
                    request_limiter=req_lim,
                    raw_book_content=book_content,
                    book_meta=cached_gb_meta,
                    sett=sett,
                    time_started=now
                )
    print(s)
    # embeddings = await create_embeddings_async(embed_client=sett.get_async_emb_client(), 
    #                                         model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
    #                                         inp_batches=[["hej", "med", "dig"]],
    #                                         tok_limiter=token_lim,
    #                                         req_limiter=req_lim
    #                                         )


if __name__ == "__main__":      # Don't run when imported via import statement
    asyncio.run(_try())

    