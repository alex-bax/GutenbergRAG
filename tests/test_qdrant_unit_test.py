import os
import uuid
from typing import AsyncGenerator
import numpy as np
import pytest
import random
from settings import Settings, get_settings  
from db.qdrant_vector_store import QdrantVectorStore
from db.vector_store_abstract import AsyncVectorStore
from models.vector_db_model import UploadChunk, EmbeddingVec

# Only run these tests if explicitly enabled.
# e.g.: RUN_QDRANT_TESTS=1 pytest tests/integration/test_qdrant_vector_store.py
pytestmark = pytest.mark.anyio  # all tests in this module are async

if not os.getenv("RUN_QDRANT_TESTS") and os.getenv("RUN_QDRANT_TESTS") == 1:
    pytest.skip(
        "Set RUN_QDRANT_TESTS=1 to run Qdrant integration tests",
        allow_module_level=True,
    )

# --- Helpers

TEST_CONTENT_FRANKENSTEIN = [
    f"discovered them. Felix soon learned that the treacherous Turk, for whom he and his family endured such unheard-of oppression, on discovering that his deliverer was thus reduced to poverty and ruin, became a traitor to good feeling and honour and had quitted Italy with his daughter, insultingly sending Felix a pittance of money to aid him, as he said, in some plan of future maintenance.",
    f"quickly round, he straightened himself out and burst into a hearty fit of laughter. “I suppose, Watson,” said he, “that you imagine that I have added opium-smoking to cocaine injections, and all the other little weaknesses on which you have favoured me with your medical views.” “I was certainly surprised to find you there.” “But not more so than I to find you.” “I came to find a friend.”"
]


def make_upload_chunks(book_id: int, chunk_content:str, sett:Settings, use_rand_embed=True) -> UploadChunk:
    """
    Helper to create some dummy UploadChunk instances.
    Uses random chunk_nr.
    """
    embed_vec = np.random.random(sett.EMBEDDING_DIM).tolist() if use_rand_embed else [0.1] * sett.EMBEDDING_DIM
    chunk = UploadChunk(
            uuid_str=str(uuid.uuid4()),
            book_id=book_id,
            book_name="Frankenstein; Or, The Modern Prometheus",
            chunk_nr=random.randint(0, 10000),
            content=chunk_content,
            content_vector=EmbeddingVec(vector=embed_vec,
                                        dim=sett.EMBEDDING_DIM)  
        )
    return chunk

def make_test_settings() -> Settings:
    """
    Create a test Settings instance suitable for talking to real Qdrant.
    """
    return get_settings(is_test=True)  # NB relies on env vars for all fields


# ---  Fixtures 

@pytest.fixture(scope="session")
def qdrant_settings() -> Settings:
    return make_test_settings()


@pytest.fixture(scope="function")
async def store(qdrant_settings: Settings) -> AsyncGenerator[AsyncVectorStore, None]:
    """
    Session-scoped store that creates the test collection once and
    deletes it after all tests.
    """
    v_store = QdrantVectorStore(
        settings=qdrant_settings,
        collection_name=qdrant_settings.active_collection,
    )

    # Ensure collection exists
    await v_store.create_missing_collection(collection_name=qdrant_settings.active_collection)

    yield v_store

    # Cleanup
    await v_store.delete_collection(collection_name=qdrant_settings.active_collection)


# --- Tests


async def test_create_missing_collection_idempotent(store: AsyncVectorStore, qdrant_settings:Settings):
    """
    Calling create_missing_collection twice should not error.
    """
    await store.create_missing_collection(collection_name=qdrant_settings.active_collection)
    await store.create_missing_collection(collection_name=qdrant_settings.active_collection)
    # If we get here without exceptions, it's fine.



async def test_upsert_chunks_and_get_missing_ids(store: AsyncVectorStore, qdrant_settings:Settings):
    """
    After upserting chunks for some book IDs, get_missing_ids_in_store
    should report only the ids that are not present.
    """
    book_ids_all = [1111, 2222, 3333]
    chunk_book_1 = make_upload_chunks(book_id=book_ids_all[0], chunk_content=TEST_CONTENT_FRANKENSTEIN[0], sett=qdrant_settings)
    chunk_book_2 = make_upload_chunks(book_id=book_ids_all[1], chunk_content=TEST_CONTENT_FRANKENSTEIN[1], sett=qdrant_settings)

    await store.upsert_chunks(chunks=[chunk_book_1, chunk_book_2])

    missing = await store.get_missing_ids_in_store(book_ids=set(book_ids_all))

    assert missing == {3333}



async def test_delete_books_removes_from_store(store: AsyncVectorStore, qdrant_settings:Settings):
    """
    After deleting a book ID, it should appear as missing.
    """
    book_ids = [4444, 5555]
    chunks_1 = make_upload_chunks(book_id=book_ids[0], chunk_content=TEST_CONTENT_FRANKENSTEIN[0], sett=qdrant_settings)
    chunks_2 = make_upload_chunks(book_id=book_ids[1], chunk_content=TEST_CONTENT_FRANKENSTEIN[0], sett=qdrant_settings)
    await store.upsert_chunks(chunks=[chunks_1, chunks_2])

    # Sanity check: nothing missing yet
    missing_before = await store.get_missing_ids_in_store(book_ids=set(book_ids))
    assert missing_before == set()

    # Delete one book
    await store.delete_books(book_ids={4444})

    # Now it should show as missing
    missing_after = await store.get_missing_ids_in_store(book_ids=set(book_ids))
    assert missing_after == {4444}



async def test_delete_collection_clears_all_data(store: AsyncVectorStore, qdrant_settings:Settings):
    """
    Deleting the collection should make all ids appear missing afterwards.
    """
    # Arrange: write some data
    book_ids = [7777, 8888]
    chunks = [make_upload_chunks(book_id=book_ids[0], chunk_content=TEST_CONTENT_FRANKENSTEIN[0], sett=qdrant_settings),
             make_upload_chunks(book_id=book_ids[1], chunk_content=TEST_CONTENT_FRANKENSTEIN[1], sett=qdrant_settings)
            ]
    await store.upsert_chunks(chunks=chunks)

    # Sanity: they are present
    missing_before = await store.get_missing_ids_in_store(book_ids=set(book_ids))
    assert missing_before == set()

    # Act: delete + recreate collection
    await store.delete_collection(collection_name=qdrant_settings.active_collection)
    await store.create_missing_collection(collection_name=qdrant_settings.active_collection)

    # Assert: now all ids are missing
    missing_after = list(await store.get_missing_ids_in_store(book_ids=set(book_ids)))
    missing_after.sort()
    assert  missing_after == book_ids
