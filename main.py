from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Annotated, Literal

from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from database import engine, SessionLocal
import psycopg2
import schema
import uvicorn

app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")

schema.Base.metadata.create_all(bind=engine)        # creates the DB tables

class BookBase(BaseModel):
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
    new_db_book = schema.Book(**book.model_dump())
    
    db.add(new_db_book)
    db.commit()
    db.refresh(new_db_book)

    return new_db_book

@prefix_router.get("/books/{book_id}", response_model=BookBase, status_code=status.HTTP_200_OK)
async def get_book(book_id:int, db:Annotated[Session, Depends(get_db)]):
    stmt = select(schema.Book).where(schema.Book.id == book_id)
    res = db.execute(stmt)
    book = res.scalars().one_or_none()
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")

    return book

# TODO: test this
@prefix_router.get("/books/", response_model=list[BookBase])
async def get_books(db:Annotated[Session, Depends(get_db)]):
    stmt = select(schema.Book)
    res = db.execute(stmt)
    books = res.all()
    return books

@prefix_router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book_id:int, db:Annotated[Session, Depends(get_db)]):
    stmt = delete(schema.Book).where(schema.Book.id == book_id)
    res = db.execute(stmt)
    db.commit()
    
    if res.rowcount == 0:       # type:ignore
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Book with id {book_id} not found")

    return

app.include_router(prefix_router)
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)