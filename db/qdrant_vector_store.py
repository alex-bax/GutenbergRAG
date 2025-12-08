import asyncio
from typing import Any, Callable
from pydantic import PrivateAttr

from settings import Settings 
from models.vector_db_model import UploadChunk, EmbeddingVec, SearchChunk, SearchPage, QDrantSearchPage

from .vector_store_abstract import AsyncVectorStore

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchText,
    Distance,
    VectorParams,
    Record,
    FacetValueHit
)

INDEXED_PAYL_FIELDS = { "chunk_nr":"integer", 
                        "book_id":"integer",
                        "book_name":"keyword",
                        "uuid_str":"uuid",
                        "content":"text",
                    }

class QdrantVectorStore(AsyncVectorStore):
    settings:Settings
    distance:Distance = Distance.COSINE
    collection_name: str
    _client: AsyncQdrantClient = PrivateAttr()

    def model_post_init(self, __context):
        self._client = AsyncQdrantClient(url=self.settings.QDRANT_SEARCH_ENDPOINT, 
                                        api_key=self.settings.QDRANT_SEARCH_KEY,
                                        verify=False)       # TODO: remove before prod and make proper fix

    def _build_must_filter(self, filters: dict[str, Any]) -> Filter:
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])
        

    async def _create_indexes(self) -> None:
        for field_name,field_type in INDEXED_PAYL_FIELDS.items():
            await self._client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
            
            
    def _points_to_search_page(self, *, points:list[Record], skip:int, limit:int, total_count:int) -> QDrantSearchPage:
        chunks: list[SearchChunk] = []
        
        for p in points:
            payload = p.payload or {}
            chunk = SearchChunk(
                search_score=-1.0,              # scroll in Qdrant doesn't give a text relevance score - using dummy value instead
                **payload)
            chunks.append(chunk)

        return QDrantSearchPage(
            chunks=chunks,
            skip_n=skip,
            top=limit,
            total_count=total_count,
        )
        

    async def _result_count_text_query(self, *, filter:Filter) -> int:
        count_res = await self._client.count(
                                collection_name=self.collection_name,
                                count_filter=filter,
                                exact=True,
                            )
        return count_res.count
    
    #TODO: make another version that just returns 1 SP!! 
    async def get_paginated_chunks_by_book_ids(self, book_ids:set[int]) -> QDrantSearchPage:
        filter = Filter(      
            must=[FieldCondition(key="book_id", match=MatchAny(any=list(book_ids))) ]
        )

        point_matches = []
        offset = None

        while True:
            points, offset = await self._client.scroll(
                                                    collection_name=self.collection_name,
                                                    scroll_filter=filter,
                                                    limit=500,
                                                    offset=offset,
                                                    with_payload=True,
                                                    with_vectors=False,
                                                )
            for p in points:
                if p.payload: 
                    point_matches.append(SearchChunk(search_score=-1.0, **p.payload))     

            if offset is None:
                break

        return QDrantSearchPage(chunks=point_matches, skip_n=0, top=1, total_count=len(point_matches))
        

    async def get_chunk_count_in_book(self, *, book_id: int) -> int:
        filter = Filter(
                must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
            )

        count = await self._result_count_text_query(filter=filter)
        return count


    async def get_chunk_by_nr(self, *, chunk_nr:int, book_id:int) -> SearchPage:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="book_id", match=MatchValue(value=book_id)),
                FieldCondition(key="chunk_nr",match=MatchValue(value=chunk_nr))
            ]
        )

        # Scroll allows retrieving all matching points with payload only
        points, next_offset = await self._client.scroll(
                                                collection_name=self.collection_name,
                                                scroll_filter=qdrant_filter,
                                                with_payload=True,
                                                with_vectors=False,
                                                limit=100,   # adjust as needed
                                            )
        assert len(points) == 1

        sp = self._points_to_search_page(points=points, 
                                         skip=0, limit=1, 
                                         total_count=1)

        return sp

    
    async def get_missing_ids_in_store(self, book_ids:set[int]) -> set[int]:
        search_page_matches = await self.get_paginated_chunks_by_book_ids(book_ids)
        existing_ids = {sp.book_id for sp in search_page_matches.chunks if sp.book_id}
        if not book_ids:
            return set()

        return book_ids - existing_ids



    # TODO: add feature to filter payload fields 
    async def paginated_search_by_text(self, *, 
                            text_query:str,
                            skip: int,  
                            limit: int = 50,
                            ) -> SearchPage:
        """
            Keyword-based search (no embeddings) in Qdrant matching on full-text field `content`.
            The search is paginated based on the 'skip' and 'limit' parameters, and returns a custom SearchPage having all payload fields.
            Args:
                skip (int): Number of chunk items to skip in the search. Similar to an 'offset'.
                limit (int): Number of chunk items to include after skipping. 
        """
        text_filter = Filter(
            must=[
                FieldCondition(
                    key="content",
                    match=MatchText(text=text_query)
                )
            ]
        )

        collected_points = []
        next_offset = None

        total_count = await self._result_count_text_query(filter=text_filter)

        while len(collected_points) < skip + limit:
            points, next_offset = await self._client.scroll(
                                            collection_name=self.collection_name,
                                            scroll_filter=text_filter,
                                            limit=limit,       # no. chunk items to return for each scroll call
                                            offset=next_offset,
                                            with_payload=True,
                                            with_vectors=False,
                                        )
            if not points:
                break  # no more results

            collected_points.extend(points)

            if next_offset is None:
                break  # reached end

        page_points = collected_points[skip:skip + limit]               # apply skip/top on the collected points

        sp = self._points_to_search_page(points=page_points, 
                                    skip=skip, limit=limit, 
                                    total_count=total_count)

        return sp


    async def create_missing_collection(self, collection_name: str) -> None:
        if not await self._client.collection_exists(collection_name=collection_name):
            await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                                        size=self.settings.EMBEDDING_DIM,
                                        distance=self.distance,
                                    ),
                )
            
            await self._create_indexes()
            
        else:
            print("Collection exists")


    async def delete_collection(self, collection_name:str) -> None:
        await self._client.delete_collection(collection_name)
    
        

    async def upsert_chunks(self, chunks: list[UploadChunk]) -> None:
        if not chunks:
            return
        
        points = [
            PointStruct(
                id=c.uuid_str,
                vector=c.content_vector.vector,
                payload={
                    "uuid_str": c.uuid_str,
                    "chunk_nr": c.chunk_nr,
                    "book_name": c.book_name,
                    "book_id": c.book_id,
                    "content": c.content,
                },
            ) for c in chunks]

        await self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search_by_embedding(self, embed_query_vector: EmbeddingVec, filter:dict[str,Any]|None, k: int=10) -> list[SearchChunk]:
        qdrant_filter = self._build_must_filter(filter) if filter else None

        results = await self._client.query_points(
                                        collection_name=self.collection_name,
                                        query=embed_query_vector.vector,
                                        limit=k,
                                        query_filter=qdrant_filter,
                                    )

        hits = [SearchChunk(search_score=p.score, **p.payload) for p in results.points if p.payload]
        
        return hits

        # async def populate_small_collection(self) -> tuple[list[GBBookMeta], str]:
        #     async_db_sess = get_async_db_sess()
        #     return await upload_missing_book_ids(book_ids=DEF_BOOK_GB_IDS_SMALL, sett=self.settings, db_sess=async_db_sess)


    async def delete_books(self, book_ids: set[int]) -> None:
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="book_id",
                        match=MatchAny(any=list(book_ids)),
                    )
                ]
            ),
        )  

    
    async def close_conn(self) -> None:
        return await self._client.close()
            
                        
    async def _get_all_unique_from_field(self, field:str) -> list[FacetValueHit]:

        resp = await self._client.facet(
                                        collection_name=self.collection_name,
                                        key=field,
                                        limit=10_00)
        
        return resp.hits

    async def get_all_unique_book_names(self) -> list[str]:
        resp_hits = await self._get_all_unique_from_field(field="book_name")
        print(f'resp_hits {resp_hits}')
        facet_vals = [hit.value for hit in resp_hits]
        fvs = [str(fv) for fv in facet_vals]
        print(fvs)
        return fvs



async def try_local() :
    from settings import get_settings
    client = QdrantVectorStore(settings=get_settings(), 
                               collection_name="gutenberg")
    # await client.initialize()
      
    # docs = [
    #         UploadChunk(
    #             uuid_str="550e8400-e29b-41d4-a716-446655440000",
    #             book_id = 42,
    #             book_name = "Moby",
    #             chunk_nr = 0,
    #             content = "Test test",
    #             content_vector=EmbeddingVec(vector=[0.0]*EmbeddingDimension.SMALL, 
    #                                         dim=EmbeddingDimension.SMALL),
    #         ),
    #     ]

    # results = await client._client.query_points(
    #                                     collection_name=client.collection_name,
    #                                     query=[0.0]*EmbeddingDimension.SMALL,
    #                                     limit=5,
    #                                     # query_filter=qdrant_filter,
    #                                 )
    # return results



if __name__ == "__main__":
    asyncio.run(try_local())