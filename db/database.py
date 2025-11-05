from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker
from settings import get_settings
import asyncio

sett = get_settings()

POSTGRES_DB_URL = f'postgresql+asyncpg://postgres:{sett.DB_PW}@localhost:{sett.DB_PORT}/{sett.DB_NAME}'
#'postgresql://postgres:Bright#1270@localhost/fastapi'

engine = create_async_engine(POSTGRES_DB_URL, echo=True) #create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base() 

