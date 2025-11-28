# from sqlalchemy import create_engine
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker
from settings import get_settings

from typing import AsyncGenerator, AsyncIterator

sett = get_settings()

POSTGRES_DB_URL = f"postgresql+asyncpg://{sett.DB_USER}:{sett.DB_PW}@aws-1-eu-north-1.pooler.supabase.com:{sett.DB_PORT}/postgres"

engine = create_async_engine(POSTGRES_DB_URL, echo=True) #create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base() 

# async def get_async_db_sess() -> AsyncIterator[AsyncSession]:
#     async with AsyncSessionLocal() as session:
#         yield session

async def get_async_db_sess() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session




@asynccontextmanager
async def get_db():
    gen = get_async_db_sess()
    session = await anext(gen)
    try:
        yield session
    finally:
        await gen.aclose()