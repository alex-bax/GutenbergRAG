from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from settings import get_settings

sett = get_settings()

SQLALCHEMY_DATABASE_URL = f'postgresql://postgres:{sett.DB_PW}@localhost:{sett.DB_PORT}/{sett.DB_NAME}'
#'postgresql://postgres:Bright#1270@localhost/fastapi'

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base() 

