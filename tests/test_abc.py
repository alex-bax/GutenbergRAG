from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, AsyncTransaction, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from db.database import Base

# Importing fastapi.Depends that is used to retrieve SQLAlchemy's session
from db.database import get_async_db_sess
# from app.api.deps import get_async_session
# Importing main FastAPI instance
from main import app

# To run async tests
pytestmark = pytest.mark.anyio

# Supply connection string
engine = create_async_engine("sqlite+aiosqlite:///:memory:")


# SQLAlchemy model for demo purposes
class Profile(DeclarativeBase):
    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str]

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
    async with connection.begin() as transaction:
        yield transaction


# Use this fixture to get SQLAlchemy's AsyncSession.
# All changes that occur in a test function are rolled back
# after function exits, even if session.commit() is called
# in inner functions
@pytest.fixture()
async def session(
    connection: AsyncConnection, transaction: AsyncTransaction
) -> AsyncGenerator[AsyncSession, None]:
    async_session = AsyncSession(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    yield async_session

    await transaction.rollback()

        
# Tests showing rollbacks between functions when using SQLAlchemy's session
async def test_create_profile(session: AsyncSession):
    existing_profiles = (await session.execute(select(Profile))).scalars().all()
    assert len(existing_profiles) == 0

    test_name = "test"
    session.add(Profile(name=test_name))
    await session.commit()

    existing_profiles = (await session.execute(select(Profile))).scalars().all()
    assert len(existing_profiles) == 1
    assert existing_profiles[0].name == test_name


async def test_rollbacks_between_functions(session: AsyncSession):
    existing_profiles = (await session.execute(select(Profile))).scalars().all()
    assert len(existing_profiles) == 0

# Use this fixture to get HTTPX's client to test API.
# All changes that occur in a test function are rolled back
# after function exits, even if session.commit() is called
# in FastAPI's application endpoints
@pytest.fixture()
async def client(
    connection: AsyncConnection, transaction: AsyncTransaction
) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_async_session() -> AsyncGenerator[AsyncSession, None]:
        async_session = AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
        )
        async with async_session:
            yield async_session
    
    # Here you have to override the dependency that is used in FastAPI's
    # endpoints to get SQLAlchemy's AsyncSession. In my case, it is
    # get_async_session
    app.dependency_overrides[get_async_db_sess] = override_get_async_session
    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://")
    del app.dependency_overrides[get_async_db_sess]

    await transaction.rollback()


# Tests showing rollbacks between functions when using API client
async def test_api_create_profile(client: AsyncClient):
    test_name = "test"
    async with client as ac:
        response = await ac.post(
            # "/api/profiles",
            "/v1/books/",
            json={
                "id": 0,
                "gb_id": 0,
                "title": "string",
                "lang": "en",
                "authors": "string"
                },
        )
        created_profile_id = response.json()["id"]

        response = await ac.get(
            "/api/profiles",
        )
        assert response.status_code == 200
        assert len(response.json()) == 1
        
        response = await ac.get(
            f"/api/profiles/{created_profile_id}",
        )
        assert response.status_code == 200
        assert response.json()["id"] == created_profile_id
        assert response.json()["name"] == test_name


async def test_client_rollbacks(client: AsyncClient):
    async with client as ac:
        response = await ac.get(
            "/api/profiles",
        )
        assert len(response.json()) == 0