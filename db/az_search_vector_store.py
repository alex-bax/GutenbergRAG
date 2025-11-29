
from constants import DEF_BOOK_GB_IDS_SMALL
from db.vector_store_abstract import AsyncVectorStore
from ingestion.book_loader import upload_missing_book_ids
from models.api_response_model import GBBookMeta
from settings import Settings
from models.vector_db_model import SearchChunk, SearchPage
from azure.search.documents.aio import SearchClient, AsyncSearchItemPaged
from azure.core.async_paging import AsyncPageIterator
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
)
from typing import Sequence, Any, cast
from azure.core.credentials import AzureKeyCredential
from pydantic import PrivateAttr
from azure.search.documents.models import VectorizedQuery
from models.vector_db_model import EmbeddingVec, UploadChunk

ALL_COLLECTION_FIELDS = ["uuid_str", "chunk_nr", "book_name", "book_id", "content"]

class AzSearchVectorStore(AsyncVectorStore):
    settings:Settings
    _search_client:SearchClient = PrivateAttr()     # not used for validation
    _index_client:SearchIndexClient = PrivateAttr()


    async def model_post_init(self, __context):
        # Run after Pydantic validates input and creates the object
        self._search_client = SearchClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                        index_name=self.settings.active_collection,
                                        credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY)
                                    )

        self._index_client = SearchIndexClient(endpoint=self.settings.AZURE_SEARCH_ENDPOINT,
                                            credential=AzureKeyCredential(self.settings.AZURE_SEARCH_KEY))
        
    def _dict_to_search_page(self, d:dict[Any,Any]) -> SearchChunk:
        return SearchChunk(
                uuid_str=d.get("uuid_str"),
                chunk_nr=d.get("chunk_nr"),
                book_name=d.get("book_name"),
                book_id=d.get("book_id"),
                content=d.get("content"),
                search_score=d["@search.score"],  # always given by Azure
            )
        

    async def _items_to_search_page(self, *, items:AsyncSearchItemPaged[dict[Any, Any]], 
                                    skip:int, limit:int, total_count:int|None=None) -> SearchPage:
        chunks = []

        async for r in items:
            payload = r  # each result is a dict-like object
            chunks.append(
                self._dict_to_search_page(payload)
            )
        total_count_ = total_count if total_count is not None else len(chunks) 
        return SearchPage(chunks=chunks, skip_n=skip, top=limit, total_count=total_count_)        


    async def upsert_chunks(self, chunks: list[UploadChunk]) -> None:
        docs = [chunk.to_dict() for chunk in chunks]
        
        for docs in docs:
            await self._search_client.upload_documents(docs)        #type:ignore


    async def _scroll_chunks_by_filter(
                self,*,
                filter_expr:str, 
                continuation_token: str | None = None,
            ) -> tuple[list[SearchChunk], str | None, int]:
        """
        Returns ONE page of results plus a continuation token.
        Call again with the returned continuation_token to get the next page.
        """

        # search returns AsyncSearchItemPaged[Dict]
        results = await self._search_client.search(
                                                search_text="*",
                                                filter=filter_expr,
                                                query_type="simple",
                                                include_total_count=True,
                                                select=ALL_COLLECTION_FIELDS,
                                            )
        total_count = await results.get_count()  

        page_iter: AsyncPageIterator[dict[str, Any]] = cast(AsyncPageIterator[dict[str, Any]],
                                                            results.by_page(continuation_token=continuation_token),
                                                        )
        chunks: list[SearchChunk] = []
        next_token: str | None = None

        # Take exactly ONE page starting at continuation_token
        async for page in page_iter:
            async for r in page:
                payload = r  # r is a dict-like SearchDocument
                chunks.append(
                    self._dict_to_search_page(payload)
                )

            next_token = page_iter.continuation_token   
            break  # only one page per call

        # page = SearchPage(
        #     chunks=chunks,
        #     # skip_n/top not applicable for token-based paging, but used as best as possible
        #     skip_n=0,
        #     top=len(chunks),
        #     total_count=total_count,
        # )

        return chunks, next_token, total_count


    async def get_chunk_by_nr(self, chunk_nr:int, book_id:int) -> SearchPage:
        filter_expr = f"book_id eq {book_id} and chunk_nr eq {chunk_nr}" 
        
        resp = await self._search_client.search(                
                                        query_type="simple",
                                        search_text="*",
                                        filter=filter_expr,
                                        facets=ALL_COLLECTION_FIELDS,
                                        include_total_count=True,
                                        top=1000
                                    )
        
        sp = await self._items_to_search_page(items=resp, skip=0, limit=1)
        return sp


    async def get_chunk_count_in_book(self, book_id:int) -> int:
        resp = await self._search_client.search(                
                                        query_type="simple",
                                        search_text="*",
                                        filter=f"book_id eq {book_id}",
                                        facets=ALL_COLLECTION_FIELDS,
                                        include_total_count=True,
                                        top=1000        # MAX
                                    )
        return await resp.get_count()


    async def get_missing_ids_in_store(self, book_ids: set[int]) -> set[int]:
        search_page_matches = await self.get_paginated_chunks_by_book_ids(book_ids=book_ids)
        existing_ids = {sp.book_id for sp in search_page_matches.chunks if sp.book_id}
        
        missing_book_ids = set(book_ids) - set(existing_ids)
        return missing_book_ids  

        
    async def get_paginated_chunks_by_book_ids(self, *, book_ids:set[int]) -> SearchPage:
        filter_expr = " or ".join([f"book_id eq {b_id}" for b_id in book_ids])
        # resp = await self._search_client.search(                
        #             query_type="simple",
        #             search_text="*",
        #             filter=filter_expr,
        #             # facets=["book_name", "book_id"],        # search using facets
        #             facets=ALL_COLLECTION_FIELDS,
        #             include_total_count=True,
        #             top=1000
        #         )
        all_chunks: list[SearchChunk] = []
        while True:
            chunks, token, total_count = await self._scroll_chunks_by_filter(
                                                    filter_expr=filter_expr,
                                                    continuation_token=token,
                                                )
            all_chunks.extend(chunks)

            if token is None:
                break  # no more pages
        
        # found_book_ids = [SearchChunk(**f["value"]) for f in resp.get_facets()["book_id"]] # type:ignore
        sp = SearchPage(chunks=all_chunks, skip_n=0, top=l, total_count=total_count)
        return sp


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


    async def delete_collection(self, collection_name:str) -> None:
        await self._index_client.delete_index(collection_name)


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


  
