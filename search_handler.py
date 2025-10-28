import os, uuid
from pathlib import Path
from dotenv import load_dotenv

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
from preprocess_book import make_slug_book_key, extract_txt, limiter_create_embeddings, batch_texts_by_tokens
from chunking import fixed_size_chunks
from settings import get_settings
from data_classes.vector_db import EmbeddingVec, ChapterDBItem

# TODO: use Pydantic Settings obj

def _get_index_fields() -> list[SearchField]:
    return [
            SimpleField(name="id_str", type=SearchFieldDataType.String, key=True),
            SimpleField(name="book_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="book_key", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,                                   
                vector_search_dimensions=EmbeddingDimension.SMALL,      # NB must match with embedding model dimension
                vector_search_profile_name="vprofile"
            ),
        ]


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


def is_book_in_index(*, search_client:SearchClient, book_key:str) :
    resp = search_client.search(                # search using facets
                query_type="simple",
                search_text="*",
                filter=f"book_key eq '{book_key}'",
                facets=["book_name", "book_key"],
                top=1,
            )
    
    return any(True for _ in list(resp))   # type:ignore


def upload_to_index(*, search_client:SearchClient, 
                    embed_client:AzureOpenAI, 
                    book:dict[str,str],
                    token_limiter:Limiter,
                    request_limiter:Limiter) -> list[ChapterDBItem]:
    sett = get_settings()
    raw_book_str = download_or_load_from_cache(book_key=book["book_key"], url=book["url"])
    
    book_str = extract_txt(raw_book=raw_book_str) #extract_chapters(book_txt=book)
    if len(book_str) == 0:      
        print(f'INFO No book string extracted --- skip {book["title"]}')
        return []

    docs:list[dict] = []
    vector_items_added = []

    chunks = fixed_size_chunks(text=book_str)
    batches = batch_texts_by_tokens(texts=chunks)

    # print(f'{len(chunks)} # txts with lens {[len(ch) for ch in chunks]}')
    # embeddings = create_embeddings(embed_client=embed_client, 
    #                                             model_deployed="text-embedding-3-small",
    #                                             texts=chunks)

    embeddings = limiter_create_embeddings(embed_client=embed_client, 
                                            model_deployed=sett.EMBED_MODEL_DEPOYED,
                                            inp_batches=batches,
                                            tok_limiter=token_limiter,
                                            req_limiter=request_limiter
                                            )
                    
        
    assert len(chunks) == len(embeddings)

    for i, (chunk, emb_vec) in enumerate(zip(chunks, embeddings)):
        chapter_item = ChapterDBItem(
            id_str= str(uuid.uuid4()),
            book_name= book["title"],
            book_key= book["book_key"],
            chunk_id= i,
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

    chapter_item = ChapterDBItem(
        id_str="123e4567-e89b-12d3-a456-426614174000",
        book_name="The Great Adventure",
        book_key="the-great-adventure",
        chunk_id=1,
        # chapter_title="Chapter 1: The Beginning",
        content="It was a bright cold day in April, and the clocks were striking thirteen.",
        content_vector=EmbeddingVec(
            vector=[0.42]*EmbeddingDimension.SMALL,  # Must match the dim size
            dim=EmbeddingDimension.SMALL
        )
    )

    print(chapter_item)

    # sett = get_settings()
    # AZURE_SEARCH_ENDPOINT = sett.AZURE_SEARCH_ENDPOINT
    # AZURE_SEARCH_KEY = sett.AZURE_SEARCH_KEY
    
    # dummy_ch_content = ["Call me Ishmael. Some years ago—never mind how long precisely..."]

    # dummy_doc = {
    #     "id": "test-1",
    #     "book": "Moby-Dick",
    #     "chapter": "CHAPTER 1. Loomings.",
    #     "chunk_id": 0,
    #     "content": dummy_ch_content,
    #     "contentVector": [0.0]*3072  # placeholder—replace with a real embedding
    # }

    # az_key = AzureKeyCredential(AZURE_SEARCH_KEY)

    # # index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=az_key)
    # # create_missing_search_index(search_index_client=index_client)
    
    # search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name="moby", credential=az_key)
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

    # # book_key = make_slug_book_key(title="Moby-Dick", gutenberg_id=42, author="Herman Melville", lang="en")

    # # upload_to_index(search_client=search_client, 
    # #                 book_url="",
    # #                 embed_client=emb_client,
    # #                 book_key=book_key)

    

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
    
    # # for r in results:
    # #     print(r["chapter"], r["@search.score"])