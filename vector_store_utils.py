import uuid
import asyncio
from azure.search.documents import SearchClient

from pyrate_limiter import Limiter
from openai import AsyncAzureOpenAI

from constants import CHUNK_SIZE

from ingestion.preprocess_book import clean_headers 
from ingestion.chunking import fixed_size_chunking
from settings import get_settings
from embedding_pipeline import batch_texts_by_tokens, create_embeddings_async

from models.api_response_model import GBBookMeta
from models.vector_db_model import SearchChunk, SearchPage, UploadChunk
from db.vector_store_abstract import AsyncVectorStore

# def _get_semantinc_search_settings() -> SemanticSearch:
#     return SemanticSearch(
#             configurations=[
#                 SemanticConfiguration(
#                     name="default",
#                     prioritized_fields=SemanticPrioritizedFields(
#                         content_fields=[SemanticField(field_name="content")]
#                     )
#                 )
#             ])


def paginated_search(*, search_client:SearchClient, q:str="", skip:int, top:int, select_fields:str|None): #-> list[SearchPage]:
    results = search_client.search(
        search_text=q,   # "" gets all
        include_total_count=True,
        select=select_fields.split(",") if select_fields else None,
        skip=skip,
        top=top
    )
    total = results.get_count()
    results_as_dicts:list[dict] = list(results)
    search_items = [SearchChunk(**page) for page in results_as_dicts]
    page = SearchPage(items=search_items, skip_n=skip, top=top, total_count=total)

    return page    # can safely do this (load into memory) since top and skip are limited via api params


def _split_by_size(data: list, chunk_size: int) -> list[list[UploadChunk]]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


async def upload_to_index_async(*, vec_store:AsyncVectorStore, 
                                embed_client:AsyncAzureOpenAI, 
                                token_limiter:Limiter,
                                request_limiter:Limiter,
                                raw_book_content: str,
                                book_meta: GBBookMeta,
                    ) -> list[UploadChunk]:
    sett = get_settings()

    book_str = clean_headers(raw_book=raw_book_content) 
    if len(book_str) == 0:      
        print(f'** INFO No book content str extracted --- skipping {book_meta.title}')
        return []

    docs:list[dict] = []
    vector_items_added = []

    chunks = fixed_size_chunking(text=book_str, chunk_size=CHUNK_SIZE)
    batches = batch_texts_by_tokens(texts=chunks)

    embeddings = await create_embeddings_async(embed_client=embed_client, 
                                            model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
                                            inp_batches=batches,
                                            tok_limiter=token_limiter,
                                            req_limiter=request_limiter
                                            )
        
    assert len(chunks) == len(embeddings)

    for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
        chapter_item = UploadChunk(
                            uuid_str= str(uuid.uuid4()),
                            book_name= book_meta.title,
                            book_id = book_meta.id,
                            chunk_nr= i,
                            content= chunk,
                            content_vector= emb_vec
                        )
        
        docs.append(chapter_item.to_dict())
        vector_items_added.append(chapter_item)

    docs_splitted = _split_by_size(data=docs, chunk_size=CHUNK_SIZE)
    for doc_chunks in docs_splitted:
        await vec_store.upsert(chunks=doc_chunks)

    return vector_items_added


async def _local_try():
    sett = get_settings()
    req_lim, token_lim = sett.make_limiters()

    embeddings = await create_embeddings_async(embed_client=sett.get_emb_client(), 
                                            model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
                                            inp_batches=[["hej", "med", "dig"]],
                                            tok_limiter=token_lim,
                                            req_limiter=req_lim
                                            )


if __name__ == "__main__":      # Don't run when imported via import statement
    asyncio.run(_local_try())

    