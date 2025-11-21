import asyncio
from typing import Any, Sequence
from pydantic import PrivateAttr
from settings import Settings 
from constants import EmbeddingDimension
from models.vector_db_model import UploadChunk, EmbeddingVec, SearchChunk
# from vector_store_abstract import AsyncVectorStore
from .vector_store_abstract import AsyncVectorStore

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Distance,
    VectorParams,
    PointIdsList
)
class QdrantVectorStore(AsyncVectorStore):
    settings:Settings
    distance:Distance = Distance.COSINE
    collection_name: str
    _client: AsyncQdrantClient = PrivateAttr()

    def model_post_init(self, __context):
        self._client = AsyncQdrantClient(url=self.settings.QDRANT_SEARCH_ENDPOINT, 
                                        api_key=self.settings.QDRANT_SEARCH_KEY)

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])
        
    
    async def initialize(self):
        await self.create_missing_collection(self.collection_name)

    async def _create_indexes(self) -> None:
        await self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="chunk_nr",
                        field_schema="integer",
                    )
        
        await self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="book_id",
                        field_schema="integer",
                    )
        
        await self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="book_name",
                        field_schema="keyword",
                    )
        
        await self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="uuid_str",
                        field_schema="uuid",
                    )


    # TODO: do this during initialization
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
        

    async def upsert(self, chunks: list[UploadChunk]) -> None:
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

    async def search_by_embedding(self, query_embed_vec: EmbeddingVec, filter:dict[str,Any], k: int=10) -> list[SearchChunk]:
        qdrant_filter = self._build_filter(filter)

        results = await self._client.query_points(
                                        collection_name=self.collection_name,
                                        query=query_embed_vec.vector,
                                        limit=k,
                                        query_filter=qdrant_filter,
                                    )

        hits = [SearchChunk(search_score=p.score, **p.payload) for p in results.points if p.payload]
        
        return hits


    async def delete_books(self, book_ids: Sequence[int]) -> None:
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
    await client.initialize()

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

    # qdrant_filter = client._build_filter({"book_id":42})

    results = await client._client.query_points(
                                        collection_name=client.collection_name,
                                        query=[0.0]*EmbeddingDimension.SMALL,
                                        limit=5,
                                        # query_filter=qdrant_filter,
                                    )
    return results
    # await client.upsert(
    #             chunks=docs,
    #         )


if __name__ == "__main__":
    asyncio.run(try_local())