from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Annotated, Literal

from sqlalchemy.orm import Session
from database import engine, SessionLocal
import psycopg2
import schema


app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")


schema.Base.metadata.create_all(bind=engine)        # creates the DB tables

class BookBase(BaseModel):
    title: str
    authors: str #list[str]
    lang:Literal["en", "da", "de"] = Field(..., description="Language the book is written")



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@prefix_router.post("/books/", response_model=BookBase, status_code=status.HTTP_201_CREATED)
async def create_book(book:BookBase, db:Annotated[Session, Depends(get_db)]):
    new_db_book = schema.Book(**book.model_dump(), slug_key="hej")
    
    db.add(new_db_book)
    db.commit()
    db.refresh(new_db_book)

    return new_db_book


app.include_router(prefix_router)
    
