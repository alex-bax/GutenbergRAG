
from constants import DEF_BOOK_GB_IDS_SMALL
from db.vector_store_abstract import AsyncVectorStore
from ingestion.book_loader import index_upload_missing_book_ids
from models.api_response_model import GBBookMeta
from settings import Settings
from models.vector_db_model import SearchChunk, SearchPage
from azure.search.documents.aio import SearchClient, AsyncSearchItemPaged
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
)
from typing import Sequence
from azure.core.credentials import AzureKeyCredential
from pydantic import PrivateAttr
from azure.search.documents.models import VectorizedQuery
from models.vector_db_model import EmbeddingVec, UploadChunk

class AzSearchVectorStore(AsyncVectorStore):
    settings:Settings
    _search_client:SearchClient = PrivateAttr()     # not used for validation
    _index_client:SearchIndexClient = PrivateAttr()

    async def model_post_init(self, __context):
        # Run after Pydantic validates input and creates the object
        self._search_client = SearchClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                        index_name=self.settings.INDEX_NAME,
                                        credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY)
                                    )

        self._index_client = SearchIndexClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                            credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY))
        

    async def upsert_chunks(self, chunks: list[UploadChunk]) -> None:
        docs = [chunk.to_dict() for chunk in chunks]
        
        for docs in docs:
            await self._search_client.upload_documents(docs)        #type:ignore


    async def get_missing_ids(self, book_ids: set[int]) -> set[int]:
        filter_expr = " or ".join([f"book_id eq {b_id}" for b_id in book_ids])
        resp = self._search_client.search(                
                    query_type="simple",
                    search_text="*",
                    filter=filter_expr,
                    facets=["book_name", "book_id"],        # search using facets
                    top=len(book_ids),
                )
        
        found_book_ids = [f["value"] for f in resp.get_facets()["book_id"]] # type:ignore
        missing_book_ids = set(book_ids) - set(found_book_ids)
        
        return missing_book_ids  

        
    async def search_by_embedding(
            self, 
            embed_query_vector:EmbeddingVec,
            k: int = 10,
        ) -> list[SearchChunk]:
        """
        Returns a list of hits (each hit is a dict with at least: id, score).
        """
        vec_q = VectorizedQuery(vector=embed_query_vector.vector, k_nearest_neighbors=k, fields="book_name, book_id, content_vector")

        results:AsyncSearchItemPaged = await self._search_client.search(
                                                vector_queries=[vec_q],
                                                top=k,
                                            )
        hits=[]
        
        async for r in results:
            chunk = SearchChunk(search_score=r["@search.score"], 
                                uuid_str=r["uuid_str"],
                                chunk_nr=r["chunk_nr"],
                                book_id=r["book_id"],
                                content=r["content"],
                                book_name=r["book_name"],
                            )
            hits.append(chunk)
        
        return hits


    async def delete_books(self, book_ids: Sequence[int]) -> None:
        doc_dicts = [{"book_id": b_id} for b_id in book_ids]
        await self._search_client.delete_documents(doc_dicts)


    async def paginated_search_by_text(self, *, 
                                text_query:str,
                                skip: int,  
                                limit: int = 50,
                            ) -> SearchPage:
        
        results = await self._search_client.search(
                            search_text=text_query,   # "" gets all
                            include_total_count=True,
                            select=None,        # As None returns all fields
                            skip=skip,
                            top=limit
                        )
        total = await results.get_count()
        
        search_items = [SearchChunk(*page, search_score=page["@search.score"]) async for page in results]

        return SearchPage(chunks=search_items, 
                          skip_n=skip, top=limit, 
                          total_count=total)
    
    async def populate_small_collection(self) -> list[GBBookMeta]:
        return await index_upload_missing_book_ids(book_ids=DEF_BOOK_GB_IDS_SMALL, sett=self.settings)

    
    def _get_index_fields(self) -> list[SearchField]:
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
                    vector_search_dimensions=self.settings.EMBEDDING_DIM,      # NB must match with embedding model dimension
                    vector_search_profile_name="vprofile"
                ))
        
        return index_fields


    def _get_vector_fields(self, vector_search_alg_name="hnsw") -> VectorSearch:
        vector_search_alg = HnswAlgorithmConfiguration(name=vector_search_alg_name) if vector_search_alg_name == "hnsw" else ExhaustiveKnnAlgorithmConfiguration(name=vector_search_alg_name)
        
        return VectorSearch(
            algorithms=[vector_search_alg],
            profiles=[VectorSearchProfile(name="vprofile",
                                        algorithm_configuration_name=vector_search_alg_name)]
        )


    async def create_missing_collection(self, collection_name:str) -> None:
        """Collection in Azure lingo for Search Index"""
        all_indexes = [idx.name async for idx in self._index_client.list_indexes()]

        if collection_name not in all_indexes:
            new_index = SearchIndex(
                name=collection_name,
                fields=self._get_index_fields(),
                vector_search=self._get_vector_fields(),
                # semantic_search=_get_semantinc_search_settings()
            )

            self._index_client.create_index(index=new_index)
            print(f"Created index: {collection_name}")
        else:
            print(f"Index '{collection_name}' already created")


    
