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

from load_book import download_or_load_from_cache
from preprocess_book import extract_chapters, tiktoken_chunks


# TODO: use Pydantic Settings obj
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]

SMALL_EMBEDDING_VECTOR_SIZE = 1536
MAX_TOKENS = 600
OVERLAP = 60

def _get_index_fields() -> list[SearchField]:
    return [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="book", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chapter", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,                                   
                vector_search_dimensions=SMALL_EMBEDDING_VECTOR_SIZE,      # NB must match with embedding model dimension
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
        print(f"Index {book_index_name} already created")


# TODO: make env var more elegant, instead of passing

def is_book_in_index(*, search_client:SearchClient, book_name:str) -> bool:
    resp = search_client.search(
                search_text="*",
                filter="book eq 'Moby-Dick'",
                select=["id", "book", "chapter", "chunk_id"],  # limit payload
                top=5
            )
    
    book_count = resp.get_count()
    assert book_count < 1, f"ERR - Too many books added: Found more than 1 book named {book_name}"
    
    return book_count == 1


def upload_to_index(*, search_client:SearchClient) -> None:
    book_p = Path("books", "moby.txt")
    book = download_or_load_from_cache(book_path=book_p)
    chapters = extract_chapters(book_txt=book)

    docs = []
    ch_chunks = []

    for i, (ch_num_k, ch_content_v) in enumerate(chapters.items(), start=1):
        ch_chunks.append(tiktoken_chunks(txt=ch_content_v['content'], 
                                         max_tokens=MAX_TOKENS, 
                                         overlap=OVERLAP))
        
        vecs = (ch_chunks)
        for i, (chunk, vec) in enumerate(zip(ch_chunks, vecs)):
            docs.append({
                "id": str(uuid.uuid4()),
                "book": "Moby-Dick",
                "chapter": ch_content_v['chapter_title'],
                "chunk_id": i,
                "content": chunk,
                "contentVector": vec
            })

        # upload in batches of ~100 to keep payload small
        if len(docs) >= 100:
            search_client.upload_documents(docs)
            docs.clear()



if __name__ == "__main__":      # Don't run when imported via import statement
    load_dotenv()
    dummy_ch_content = ["Call me Ishmael. Some years ago—never mind how long precisely..."]

    dummy_doc = {
        "id": "test-1",
        "book": "Moby-Dick",
        "chapter": "CHAPTER 1. Loomings.",
        "chunk_id": 0,
        "content": dummy_ch_content,
        "contentVector": [0.0]*3072  # placeholder—replace with a real embedding
    }

    az_key = AzureKeyCredential(AZURE_SEARCH_KEY)

    index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=az_key)
    create_missing_search_index(search_index_client=index_client)

    index = index_client.get_index("moby")
    # See the field definitions
    for field in index.fields:
        print(f"Name: {field.name}")
        print(f"  Type: {field.type}")
        print(f"  Searchable: {getattr(field, 'searchable', False)}")
        print(f"  Filterable: {getattr(field, 'filterable', False)}")
        print(f"  Sortable: {getattr(field, 'sortable', False)}")
        print(f"  Facetable: {getattr(field, 'facetable', False)}")
        print(f"  Retrievable: {getattr(field, 'retrievable', False)}")
        print()

    
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name="moby", credential=az_key)
    resp = search_client.search(
                search_text="*",
                filter="book eq 'Moby-Dick'",
                select=["id", "book", "chapter", "chunk_id"],  # limit payload
                top=5
            )
    for doc in resp:
        print(doc["id"], doc["chapter"], doc["chunk_id"])
    
    # upload_book_to_index(search_client=search_client)
    # search_client.upload_documents(documents=[dummy_doc])

    # # Query (hybrid + semantic)
    # results = search_client.search(
    #     search_text="Why does Ishmael go to sea?",
    #     top=5,
    #     # query_type="semantic",
    #     # semantic_configuration_name="default"
    # )
    
    # for r in results:
    #     print(r["chapter"], r["@search.score"])