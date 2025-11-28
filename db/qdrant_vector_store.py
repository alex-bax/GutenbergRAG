import asyncio
from typing import Any, Sequence
from pydantic import PrivateAttr
from db.database import get_async_db_sess
from settings import Settings 
from constants import EmbeddingDimension 
from ingestion.book_loader import upload_missing_book_ids
from models.vector_db_model import UploadChunk, EmbeddingVec, SearchChunk, SearchPage
from models.api_response_model import GBBookMeta
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
    PointIdsList
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

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])
        
    # async def initialize(self):
    #     await self.create_missing_collection(self.collection_name)
    #     await self.populate_small_collection()


    async def result_count_text_query(self, *, text_match_filter:Filter) -> int:
        count_res = await self._client.count(
                                collection_name=self.collection_name,
                                count_filter=text_match_filter,
                                exact=True,
                            )
        return count_res.count
    

    async def _create_indexes(self) -> None:
        for field_name,field_type in INDEXED_PAYL_FIELDS.items():
            await self._client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
        

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

        total_count = await self.result_count_text_query(text_match_filter=text_filter)

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

        chunks: list[SearchChunk] = []
        for p in page_points:
            payload = p.payload or {}
            chunk = SearchChunk(
                search_score=-1.0,              # scroll in Qdrant doesn't give a text relevance score - using dummy value instead
                **payload)
            chunks.append(chunk)

        # 6) Return your SearchPage
        return SearchPage(
            chunks=chunks,
            skip_n=skip,
            top=limit,
            total_count=total_count,
        )


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
        qdrant_filter = self._build_filter(filter) if filter else None

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
                points_selector=PointIdsList(points=list(book_ids)),

                filter=Filter(      # payload field book_id IN book_ids
                        must=[
                            FieldCondition(
                                key="book_id",
                                match=MatchAny(any=list(book_ids)),
                            ),
                        ],
                    )
                )
        

    async def get_missing_ids(self, book_ids:set[int]) -> set[int]:
        if not book_ids:
            return set()

        q_filter = Filter(      # filter: return only points where book_id is in our provided list
            must=[
                FieldCondition(
                    key="book_id",
                    match=MatchAny(any=list(book_ids)),
                )
            ]
        )

        existing_ids: set[int] = set()
        offset = None

        # Scroll `book_id` to reduce payload load
        while True:
            points, offset = await self._client.scroll(
                                                    collection_name=self.collection_name,
                                                    scroll_filter=q_filter,
                                                    limit=500,
                                                    offset=offset,
                                                    with_payload=True,
                                                    with_vectors=False,
                                                )
            for p in points:
                if p.payload: 
                    existing_ids.add(p.payload["book_id"])      

            if offset is None:
                break

        return book_ids - existing_ids



async def try_local() :
    from settings import get_settings
    client = QdrantVectorStore(settings=get_settings(), 
                               collection_name="gutenberg")
    # await client.initialize()

    docs = [
            UploadChunk(
                uuid_str="550e8400-e29b-41d4-a716-446655440000",
                book_id = 42,
                book_name = "Moby",
                chunk_nr = 0,
                content = "Test test",
                content_vector=EmbeddingVec(vector=[0.0]*EmbeddingDimension.SMALL, 
                                            dim=EmbeddingDimension.SMALL),
            ),
        ]

    results = await client._client.query_points(
                                        collection_name=client.collection_name,
                                        query=[0.0]*EmbeddingDimension.SMALL,
                                        limit=5,
                                        # query_filter=qdrant_filter,
                                    )
    return results



if __name__ == "__main__":
    asyncio.run(try_local())