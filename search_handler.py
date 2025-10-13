import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
    SemanticSearch, SemanticField, SemanticConfiguration, SemanticPrioritizedFields # type: ignore
)


EMBEDDING_MODEL_SIZE = 3072

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
                vector_search_dimensions=EMBEDDING_MODEL_SIZE,      # NB must match with embedding model dimension
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


def _get_semantinc_search_settings() -> SemanticSearch:
    return SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")]
                    )
                )
            ])


def create_search_index(*, book_index_name="moby", search_index_client:SearchIndexClient) -> None:
    indexes = [idx.name for idx in search_index_client.list_indexes()]

    if book_index_name not in indexes:
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



if __name__ == "__main__":      # Don't run when imported via import statement
    load_dotenv()

    doc = {
        "id": "test-1",
        "book": "Moby-Dick",
        "chapter": "CHAPTER 1. Loomings.",
        "chunk_id": 0,
        "content": "Call me Ishmael. Some years ago—never mind how long precisely...",
        "contentVector": [0.0]*3072  # placeholder—replace with a real embedding
    }

    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    key = os.environ["AZURE_SEARCH_KEY"]

    search_client = SearchClient(endpoint=endpoint, index_name="moby", credential=AzureKeyCredential(key))
    search_index_client = SearchIndexClient(endpoint, AzureKeyCredential(key))

    create_search_index(book_index_name="moby", search_index_client=search_index_client)
    search_client.upload_documents(documents=[doc])

    # Query (hybrid + semantic)
    results = search_client.search(
        search_text="Why does Ishmael go to sea?",
        top=5,
        # query_type="semantic",
        # semantic_configuration_name="default"
    )
    
    for r in results:
        print(r["chapter"], r["@search.score"])