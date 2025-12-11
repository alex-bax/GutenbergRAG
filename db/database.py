from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker
from config.settings import get_settings

from typing import AsyncGenerator, AsyncIterator

sett = get_settings()

POSTGRES_DB_URL = f"postgresql+asyncpg://{sett.DB_USER}:{sett.DB_PW}@aws-1-eu-north-1.pooler.supabase.com:{sett.DB_PORT}/{sett.DB_NAME}"

engine = create_async_engine(POSTGRES_DB_URL, echo=True) 

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base() 

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