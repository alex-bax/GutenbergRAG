import uuid
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
    SemanticSearch, SemanticField, SemanticConfiguration, SemanticPrioritizedFields # type: ignore
)
from pyrate_limiter import Limiter
from openai import AzureOpenAI
from constants import CHUNK_SIZE

from constants import EmbeddingDimension
from preprocess_book import make_slug_book_key, clean_headers, create_embeddings_async, batch_texts_by_tokens
from chunking import fixed_size_chunks
from settings import get_settings

from models.api_response_model import GBBookMeta
from models.vector_db_model import VectorChunk, SearchChunk, SearchPage


def _get_index_fields() -> list[SearchField]:
    index_fields= [
            SimpleField(name="uuid_str", type=SearchFieldDataType.String, key=True),
            SimpleField(name="chunk_nr", type=SearchFieldDataType.Int32, filterable=True, facetable=True),      
            SimpleField(name="book_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="book_id", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        ]
    
    assert all(sf.name in SearchChunk.model_fields.keys() for sf in index_fields), f"Vector index fields not matching SearchItem: {SearchChunk.model_fields.keys()} != (index_fields) {index_fields} "

    index_fields.append(SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,                                   
                vector_search_dimensions=EmbeddingDimension.SMALL,      # NB must match with embedding model dimension
                vector_search_profile_name="vprofile"
            ))
    
    return index_fields


def _get_vector_search(vector_search_alg_name="hnsw") -> VectorSearch:
    vector_search_alg = HnswAlgorithmConfiguration(name=vector_search_alg_name) if vector_search_alg_name == "hnsw" else ExhaustiveKnnAlgorithmConfiguration(name=vector_search_alg_name)
    
    return VectorSearch(
        algorithms=[vector_search_alg],
        profiles=[VectorSearchProfile(name="vprofile",
                                      algorithm_configuration_name=vector_search_alg_name)]
    )


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




# def create_missing_search_index(*, book_index_name="moby", search_index_client:SearchIndexClient) -> None:
#     all_indexes = [idx.name for idx in search_index_client.list_indexes()]

#     if book_index_name not in all_indexes:
#         new_index = SearchIndex(
#             name=book_index_name,
#             fields=_get_index_fields(),
#             vector_search=_get_vector_search(),
#             # semantic_search=_get_semantinc_search_settings()
#         )

#         search_index_client.create_index(index=new_index)
#         print(f"Created index: {book_index_name}")
#     else:
#         print(f"Index '{book_index_name}' already created")


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

#TODO: write test for this
# TODO: make async?
def get_missing_books_in_index(*, search_client:SearchClient, book_ids:list[int]) -> list[int]:
    """
    Given a list of Gutenberg IDs return a list of those missing from the index
    """
    filter_expr = " or ".join([f"book_id eq {b_id}" for b_id in book_ids])
    resp = search_client.search(                
                query_type="simple",
                search_text="*",
                filter=filter_expr,
                facets=["book_name", "book_id"],        # search using facets
                top=len(book_ids),
            )
    
    found_book_ids = [f["value"] for f in resp.get_facets()["book_id"]] # type:ignore
    missing_books = list(set(book_ids) - set(found_book_ids))      
    
    return missing_books   


def _split_by_size(data: list, chunk_size: int) -> list[list]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


async def upload_to_index_async(*, search_client:SearchClient, 
                                embed_client:AzureOpenAI, 
                                token_limiter:Limiter,
                                request_limiter:Limiter,
                                raw_book_content: str,
                                book_meta: GBBookMeta,
                    ) -> list[VectorChunk]:
    sett = get_settings()

    book_str = clean_headers(raw_book=raw_book_content) 
    if len(book_str) == 0:      
        print(f'** INFO No book content str extracted --- skipping {book_meta.title}')
        return []

    docs:list[dict] = []
    vector_items_added = []

    chunks = fixed_size_chunks(text=book_str, chunk_size=CHUNK_SIZE)
    batches = batch_texts_by_tokens(texts=chunks)

    embeddings = await create_embeddings_async(embed_client=embed_client, 
                                            model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
                                            inp_batches=batches,
                                            tok_limiter=token_limiter,
                                            req_limiter=request_limiter
                                            )
        
    assert len(chunks) == len(embeddings)

    for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
        chapter_item = VectorChunk(
            uuid_str= str(uuid.uuid4()),
            book_name= book_meta.title,
            book_id = book_meta.id,
            chunk_nr= i,
            content= chunk,
            content_vector= emb_vec
        )
        
        docs.append(chapter_item.to_dict())
        vector_items_added.append(chapter_item)

    # upload after each chapter, max in batches of ~50 to keep payload small
    docs_splitted = _split_by_size(data=docs, chunk_size=CHUNK_SIZE)
    for docs in docs_splitted:
        search_client.upload_documents(docs)

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
    "moby-dick-or-the-whale_melville-herman_2701_en"

    asyncio.run(_local_try())

    