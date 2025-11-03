from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional
import psycopg2
import uvicorn

from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from db.database import engine, SessionLocal
from db.operations import select_all_books, select_book, delete_book, insert_book, select_books_like, BookNotFoundException
from db.schema import DBBook
import db.schema as schema

app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")

schema.Base.metadata.create_all(bind=engine)        # creates the DB tables

class BookBase(BaseModel):
    id: int
    title: str
    authors: str #list[str]
    lang:Literal["en", "da", "de"] = Field(..., description="Language the book is written")
    slug_key: str
    
    model_config = {"from_attributes": True}        # TODO: is this needed?

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@prefix_router.post("/books/", response_model=BookBase, status_code=status.HTTP_201_CREATED)
async def create_book(book:BookBase, db:Annotated[Session, Depends(get_db)]):
    new_db_book = DBBook(**book.model_dump())
    inserted_book = insert_book(new_db_book, db)

    return inserted_book


@prefix_router.get("/books/search", response_model=list[BookBase], status_code=status.HTTP_200_OK)
async def search_books(db:Annotated[Session, Depends(get_db)], 
                       title: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       authors: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       lang:Annotated[str|None, Query(min_length=2, max_length=2, examples=["en", "da", "nl"])] = None ):
    
    if not any([title, authors, lang]):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provide at least one filter parameter.")
    
    db_books = select_books_like(title=title, authors=authors, lang=lang, db_sess=db)

    return db_books


@prefix_router.get("/books/{book_id}", response_model=BookBase, status_code=status.HTTP_200_OK)
async def get_book(book_id:int, db:Annotated[Session, Depends(get_db)]):
    book = None
    try:
        book = select_book(book_id, db)
    except BookNotFoundException: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")

    if not book:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Book with id {book_id} empty")

    return book

# TODO: could be slow if DB is huge, use pagination instead
@prefix_router.get("/books/", response_model=list[BookBase])
async def get_books(db:Annotated[Session, Depends(get_db)]):
    books = select_all_books(db)
    return books


@prefix_router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_book(book_id:int, db:Annotated[Session, Depends(get_db)]):
    try:
        delete_book(book_id, db)
    except BookNotFoundException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'{exc}')

app.include_router(prefix_router)
    
# TODO: list all docs from a book, and paginate the results
# @prefix_router.get("")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)