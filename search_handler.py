import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
    SemanticSettings, SemanticField, SemanticConfiguration, SemanticPrioritizedFields # type: ignore
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
                searchable=False,                                   # True when used for full text search, and we don't use the content as text here, but instead the embedding vector for similarity search 
                vector_search_dimensions=EMBEDDING_MODEL_SIZE,      # match your embedding model
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


def _get_semantinc_search_settings() -> SemanticSettings:
    return SemanticSettings(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")]
                    )
                )
            ])


def create_search_index(book_index="moby") -> None:
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    key = os.environ["AZURE_SEARCH_KEY"]

    search_client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    indexes = [idx.name for idx in search_client.list_indexes()]

    if book_index not in indexes:
        index = SearchIndex(
            name=book_index,
            fields=_get_index_fields(),
            vector_search=_get_vector_search(),
            semantic_search=_get_semantinc_search_settings()
        )

