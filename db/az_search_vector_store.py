
from vector_store_abstract import AsyncVectorStore
from settings import Settings
from models.vector_db_model import ContentUploadChunk, SearchChunk
from azure.search.documents.aio import SearchClient, AsyncSearchItemPaged
from azure.search.documents.indexes import SearchIndexClient
from azure.core.paging import ItemPaged
from azure.core.credentials import AzureKeyCredential
from pydantic import PrivateAttr
from azure.search.documents.models import VectorizedQuery
from models.vector_db_model import EmbeddingVec

class AzSearchVectorStore(AsyncVectorStore):
    settings:Settings
    chunk_size:int
    _search_client:SearchClient = PrivateAttr()     # not used for validation

    async def model_post_init(self, __context):
        # Run after Pydantic validates input and creates the object
        self._search_client = SearchClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                        index_name=self.settings.INDEX_NAME,
                                        credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY)
                                    )

        self.index_client = SearchIndexClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                            credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY))
        

    async def upsert(self, chunks: list[ContentUploadChunk]) -> None:
        docs = [chunk.to_dict() for chunk in chunks]
        
        for docs in docs:
            await self._search_client.upload_documents(docs)        #type:ignore


    async def get_missing_ids(self, book_ids: list[int]) -> list[int]:
        filter_expr = " or ".join([f"book_id eq {b_id}" for b_id in book_ids])
        resp = self._search_client.search(                
                    query_type="simple",
                    search_text="*",
                    filter=filter_expr,
                    facets=["book_name", "book_id"],        # search using facets
                    top=len(book_ids),
                )
        
        found_book_ids = [f["value"] for f in resp.get_facets()["book_id"]] # type:ignore
        missing_book_ids = list(set(book_ids) - set(found_book_ids))      
        
        return missing_book_ids  

        
    async def search_chunks(
            self, embed_query_vector:EmbeddingVec,
            k: int = 10,
        ) -> list[SearchChunk]:
        """
        Returns a list of hits (each hit is a dict with at least: id, score, payload).
        """
        vec_q = VectorizedQuery(vector=embed_query_vector.vector, k_nearest_neighbors=40, fields="book_name, book_id, content_vector")

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



    async def delete(*, self, ids: list[str]) -> None:
        ...

    async def create_index(*, self, ids: list[str]) -> None:
        ...


    
