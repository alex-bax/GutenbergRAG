from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker
from config.settings import get_settings

from typing import AsyncGenerator, AsyncIterator, Callable, TypeAlias

sett = get_settings()

POSTGRES_DB_URL = f"postgresql+asyncpg://{sett.DB_USER}:{sett.DB_PW}@aws-1-eu-north-1.pooler.supabase.com:{sett.DB_PORT}/{sett.DB_NAME}"

engine = create_async_engine(POSTGRES_DB_URL, echo=True) 

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base() 

# @asynccontextmanager
# async def open_db_session() -> AsyncGenerator[AsyncSession, None]:
#     gen = get_async_db_sess()       # dependency generator
#     sess = await anext(gen)
#     try:
#         yield sess
#     finally:
#         await gen.aclose()

async def get_async_db_sess() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

DbSessionFactory: TypeAlias = Callable[[], AsyncGenerator[AsyncSession, None]]

def get_db_session_factory() -> DbSessionFactory:
    return get_async_db_sess


@asynccontextmanager
async def open_session(factory: DbSessionFactory) -> AsyncGenerator[AsyncSession, None]:
    """
    Turns a 'yielding dependency function' into an async context manager.
    """
    gen = factory()
    sess = await anext(gen)
    try:
        yield sess
    finally:
        await gen.aclose()

# async def _get_async_db_sess() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as session:
#         yield session

# @asynccontextmanager
# async def get_async_db_sess():
#     gen = _get_async_db_sess()
#     session = await anext(gen)
#     try:
#         yield session
#     finally:
#         await gen.aclose()