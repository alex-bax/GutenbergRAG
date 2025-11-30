from typing import AsyncGenerator
from constants import ID_DR_JEK_MR_H, ID_FRANKENSTEIN, VER_PREFIX
from settings import Settings, get_settings
import pytest
from fastapi import status
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, AsyncTransaction, create_async_engine

from db.database import Base
from models.api_response_model import BookMetaApiResponse, BookMetaDataResponse, GBMetaApiResponse, QueryResponseApiResponse, SearchApiResponse
# Importing fastapi.Depends that is used to retrieve SQLAlchemy's session
from db.database import get_async_db_sess
from db.operations import insert_book_db, DBBookMetaData
from main import app

### The test DB is rolled back after each test fixture

# To run async tests
pytestmark = pytest.mark.anyio

engine = create_async_engine("sqlite+aiosqlite:///:memory:")

@pytest.fixture(scope="session", autouse=True)
async def create_test_schema():
    # Create the schema (tables)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Drop the schema (tables) after the tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# Required per https://anyio.readthedocs.io/en/stable/testing.html#using-async-fixtures-with-higher-scopes
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def connection(anyio_backend) -> AsyncGenerator[AsyncConnection, None]:
    async with engine.connect() as connection:
        yield connection

        
@pytest.fixture()
async def transaction(
    connection: AsyncConnection,
) -> AsyncGenerator[AsyncTransaction, None]:
    trans = await connection.begin()
    try:
        yield trans
    finally:
        # Always rollback so DB is clean between tests
        await trans.rollback()

# Use this fixture to get SQLAlchemy's AsyncSession.
# All changes that occur in a test function are rolled back
# after function exits, even if session.commit() is called
# in inner functions
@pytest.fixture()
async def session(
    connection: AsyncConnection, transaction: AsyncTransaction) -> AsyncGenerator[AsyncSession, None]:
    async_session = AsyncSession(
                        bind=connection,
                        join_transaction_mode="create_savepoint",
                    )
    try:
        yield async_session
    finally:
        await async_session.close()  # only close the session
    # await transaction.rollback()

@pytest.fixture()
def test_settings() -> Settings:
    sett = get_settings()
    sett.is_test = True
    return sett

# Use this fixture to get HTTPX's client to test API.
# All changes that occur in a test function are rolled back
# after function exits, even if session.commit() is called
# in FastAPI's application endpoints
@pytest.fixture()
async def client(
    connection: AsyncConnection,
    test_settings:Settings,
) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_async_session() -> AsyncGenerator[AsyncSession, None]:
        async_session = AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )
        async with async_session:
            yield async_session
    
    app.dependency_overrides[get_async_db_sess] = override_get_async_session
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    test_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://")

    try:
        yield test_client
    finally:
        try:
            v_store = await test_settings.get_vector_store()
            await v_store.delete_collection(collection_name=test_settings.active_collection)
        except Exception as e:
            # Don't fail tests on cleanup
            print(f"[TEST CLEANUP WARNING] Could not delete collection: {e}")

        await test_client.aclose()
        del app.dependency_overrides[get_async_db_sess]
        del app.dependency_overrides[get_settings]
        # v_store = await test_settings.get_vector_store()
        # await v_store.delete_collection(collection_name=test_settings.active_collection)

        # await test_client.aclose()
        # del app.dependency_overrides[get_async_db_sess]


def test_settings_load_env_sanity_check(test_settings: Settings):
    # Sanity check that agent in pipeline has access to env vars
    assert test_settings.AZURE_SEARCH_ENDPOINT is not None
    assert test_settings.QDRANT_SEARCH_ENDPOINT is not None
    

# Tests showing rollbacks between functions when using API client
async def test_get_book(client: AsyncClient, session: AsyncSession):
    book_id = await insert_book_db(book=DBBookMetaData(gb_id=42, title="string", lang="en", authors="string"), 
                                   db_sess=session)
    
    async with client as ac:
        created_book_id = book_id #response.json()["id"]
        
        resp = await ac.get(
            f"/{VER_PREFIX}/books/{created_book_id}",
        )
        resp_model = BookMetaApiResponse(**resp.json())

        assert resp.status_code == status.HTTP_200_OK
        assert isinstance(resp_model.data, list) and len(resp_model.data) == 1
        assert isinstance(resp_model.data[0], BookMetaDataResponse) 
        assert resp_model.data[0].id == created_book_id


async def test_get_book_not_found_returns_404(client: AsyncClient):
    async with client as ac:
        non_existing_id = 999999  
        resp = await ac.get(f"/{VER_PREFIX}/books/{non_existing_id}")
        assert resp.status_code == status.HTTP_404_NOT_FOUND

# NB Only testing code correctness, not the LLM quality of the reponses, check eval instead

async def test_upload_1_to_index(client: AsyncClient, test_settings: Settings):
    async with client as ac:
        body = [ID_DR_JEK_MR_H]
        vec_store = await test_settings.get_vector_store()
        missing_ids_before = await vec_store.get_missing_ids_in_store(book_ids=set(body))
        assert body[0] in missing_ids_before

        resp = await ac.post(f"/{VER_PREFIX}/index", json=body)
        assert resp.status_code == status.HTTP_201_CREATED

        resp_model = GBMetaApiResponse(**resp.json())
        print(resp_model)

        missing_ids_after = await vec_store.get_missing_ids_in_store(book_ids=set(body))
        assert body[0] not in missing_ids_after 

# TODO: test this from api
# search_books

async def test_upload_same_twice_to_index_returns_422(client: AsyncClient):
    async with client as ac:
        body = [ID_DR_JEK_MR_H, ID_DR_JEK_MR_H]
        resp = await ac.post(f"/{VER_PREFIX}/index", json=body)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


async def test_upload_delete_book_index(client:AsyncClient, test_settings: Settings):
    async with client as ac:
        body = [ID_FRANKENSTEIN]
        vec_store = await test_settings.get_vector_store()
        chunk_count_before = await vec_store.get_chunk_count_in_book(book_id=body[0])

        resp = await ac.post(f"/{VER_PREFIX}/index", json=body)
        assert resp.status_code == status.HTTP_201_CREATED
        resp_model = GBMetaApiResponse(**resp.json())
        assert resp_model.data[0].id == body[0]

        resp_del = await ac.delete(f"/{VER_PREFIX}/index/{body[0]}")
        assert resp_del.status_code == status.HTTP_204_NO_CONTENT

        # Check that it's not there
        chunk_count_after = await vec_store.get_chunk_count_in_book(book_id=body[0])
        assert chunk_count_before == chunk_count_after


async def test_delete_book_index_not_found_returns_422(client:AsyncClient):
    async with client as ac:
        resp = await ac.delete(f"/{VER_PREFIX}/index/{ID_FRANKENSTEIN}")
        assert resp.status_code == status.HTTP_404_NOT_FOUND


# TODO - if possible try make parameterised for multiple top N chunks
async def test_answer_query_top_1_match(client: AsyncClient):
    async with client as ac:
        body = [ID_DR_JEK_MR_H]
        resp = await ac.post(f"/{VER_PREFIX}/index", json=body)
        assert resp.status_code == status.HTTP_201_CREATED
        gb_meta_model = GBMetaApiResponse(**resp.json())
        assert gb_meta_model.data[0].id == body[0]

        params = {
            "query": "What is the book Dr Jekyll and Mr Hyde about?",
            "top_n_matches": 5,
        }

        resp = await ac.get(f"/{VER_PREFIX}/query/", params=params)
        assert resp.status_code == status.HTTP_202_ACCEPTED

        query_resp = QueryResponseApiResponse(**resp.json())
        
        assert isinstance(query_resp.data.answer, str)
        assert query_resp.data.answer.strip() != ""
        assert len(query_resp.data.citations) > 0


async def test_show_gutenberg_book(client: AsyncClient):
    async with client as ac:
        test_id = ID_FRANKENSTEIN
        resp = await ac.get(f"/{VER_PREFIX}/books/gutenberg/{test_id}")
        assert resp.status_code == status.HTTP_200_OK

        gb_meta_model = GBMetaApiResponse(**resp.json())
        assert len(gb_meta_model.data) > 0
        assert gb_meta_model.data[0].id == test_id
        assert gb_meta_model.data[0].title.lower() == "Frankenstein; Or, The Modern Prometheus".lower()

        