from typing import Any, Sequence
from vector_store_abstract import ContentChunk, AsyncVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Document
)

class QdrantVectorStore(AsyncVectorStore):
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def _build_filter(self, filters: dict[str, Any]|None) -> Filter|None:
        if not filters:
            return None
        
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])
        

    def create_index(self, ids: list[str]) -> None:
        return super().create_index(ids)

    def upsert(self, docs: list[ContentChunk]) -> None:
        if not docs:
            return

        points = [
            PointStruct(
                id=d.id,
                vector={
                "my-bm25-vector": Document(
                    text="Recipe for baking chocolate chip cookies",
                    model="Qdrant/bm25",
                )
            },
                payload=d.payload,
            )
            for d in docs
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        qdrant_filter = self._build_filter(filters)

        results = self.client.query(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        )

        hits: list[dict[str, Any]] = []
        for r in results:
            hits.append(
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload,
                }
            )
        return hits

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )