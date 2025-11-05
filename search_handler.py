import os, uuid
from pathlib import Path
from dotenv import load_dotenv

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

from load_book import download_or_load_from_cache
from constants import EmbeddingDimension
from preprocess_book import make_slug_book_key, clean_headers, create_embeddings_async, batch_texts_by_tokens
from chunking import fixed_size_chunks
from settings import get_settings
from models.vector_db import ContentUploadChunk, SearchChunk, SearchPage

from models.api_response import GBBookMeta


def _get_index_fields() -> list[SearchField]:
    index_fields= [
            SimpleField(name="uuid_str", type=SearchFieldDataType.String, key=True),
            SimpleField(name="chunk_nr", type=SearchFieldDataType.Int32, filterable=True),      
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


def create_missing_search_index(*, book_index_name="moby", search_index_client:SearchIndexClient) -> None:
    all_indexes = [idx.name for idx in search_index_client.list_indexes()]

    if book_index_name not in all_indexes:
        new_index = SearchIndex(
            name=book_index_name,
            fields=_get_index_fields(),
            vector_search=_get_vector_search(),
            # semantic_search=_get_semantinc_search_settings()
        )

        search_index_client.create_index(index=new_index)
        print(f"Created index: {book_index_name}")
    else:
        print(f"Index '{book_index_name}' already created")


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


def is_book_in_index(*, search_client:SearchClient, book_id:int) :
    resp = search_client.search(                # search using facets
                query_type="simple",
                search_text="*",
                filter=f"book_id eq {book_id}",
                facets=["book_name", "book_id"],
                top=1,
            )
    
    return any(True for _ in list(resp))   # type:ignore


async def upload_to_index_async(*, search_client:SearchClient, 
                    embed_client:AzureOpenAI, 
                    token_limiter:Limiter,
                    request_limiter:Limiter,
                    raw_book_content: str,
                    book_meta: GBBookMeta,
                    ) -> list[ContentUploadChunk]:
    sett = get_settings()

    book_str = clean_headers(raw_book=raw_book_content) 
    if len(book_str) == 0:      
        print(f'** INFO No book content str extracted --- skipping {book_meta.title}')
        return []

    docs:list[dict] = []
    vector_items_added = []

    chunks = fixed_size_chunks(text=book_str)
    batches = batch_texts_by_tokens(texts=chunks)

    embeddings = await create_embeddings_async(embed_client=embed_client, 
                                            model_deployed=sett.EMBED_MODEL_DEPOYED,
                                            inp_batches=batches,
                                            tok_limiter=token_limiter,
                                            req_limiter=request_limiter
                                            )
        
    assert len(chunks) == len(embeddings)

    for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
        chapter_item = ContentUploadChunk(
            uuid_str= str(uuid.uuid4()),
            book_name= book_meta.title,
            book_id = book_meta.id,
            chunk_nr= i,
            content= chunk,
            content_vector= emb_vec
        )
        
        docs.append(chapter_item.to_dict())
        vector_items_added.append(chapter_item)

    # upload after each chapter, max in batches of ~100 to keep payload small
    search_client.upload_documents(docs)
    if len(docs) >= 100:
        docs.clear()

    return vector_items_added


if __name__ == "__main__":      # Don't run when imported via import statement
    load_dotenv()

    sett = get_settings()
    # dummy_ch_content = ["Call me Ishmael. Some years ago—never mind how long precisely..."]

    az_key = AzureKeyCredential(sett.AZURE_SEARCH_KEY)

    # # index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=az_key)
    # # create_missing_search_index(search_index_client=index_client)
    
    search_client = SearchClient(endpoint=sett.AZURE_SEARCH_ENDPOINT, index_name="moby", credential=az_key)
    paginated_search(search_client=search_client, top=5, skip=0, select_fields="book_name, id_str, chunk_nr")
    # # resp = search_client.search(
    # #             search_text="*",
    # #             filter="book eq 'Moby-Dick'",
    # #             select=["id", "book", "chapter", "chunk_id"],  # limit payload
    # #             top=5
    # #         )
    # # for doc in resp:
    # #     print(doc["id"], doc["chapter"], doc["chunk_id"])
    
    # # emb_client = AzureOpenAI(azure_endpoint=sett.AZ_OPENAI_EMBED_ENDPOINT,
    # #                         api_version="2024-12-01-preview",
    # #                         api_key=sett.AZ_OPENAI_EMBED_KEY)

    # resp = search_client.search(
    #             query_type="simple",
    #             search_text="*",
    #             filter="book eq 'Moby-Dick'",
    #             select=["id", "book", "chapter", "chunk_id"],  # limit payload
    #             top=0,
    #             include_total_count=True
    #         )
    
    # book_count = resp.get_count()
    # print(book_count)
    
    # # search_client.upload_documents(documents=[dummy_doc])

    # # # Query (hybrid + semantic)
    # # results = search_client.search(
    # #     search_text="Why does Ishmael go to sea?",
    # #     top=5,
    # #     # query_type="semantic",
    # #     # semantic_configuration_name="default"
    # # )
    