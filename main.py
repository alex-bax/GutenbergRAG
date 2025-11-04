from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, Optional
import psycopg2
import uvicorn, requests

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from db.database import engine, SessionLocal
from db.operations import select_all_books, select_book, delete_book, insert_book, select_books_like, select_documents_paginated, BookNotFoundException
from db.schema import DBBook
import db.schema as schema
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page, add_pagination, paginate

from search_handler import paginated_search, SearchPage
from settings import get_settings

app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")

schema.Base.metadata.create_all(bind=engine)        # creates the DB tables

class Book(BaseModel):
    id: int
    title: str
class BookBase(Book):
    lang:str = Field(..., description="Language ISO code the book is written in", examples=["en", "nl", "da"])
    slug_key: str
    authors: str #list[str]
    model_config = {"from_attributes": True}        # TODO: is this needed?


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@prefix_router.post("/books/", response_model=Book, status_code=status.HTTP_201_CREATED)
async def create_book(book:BookBase, db:Annotated[Session, Depends(get_db)]):
    new_db_book = DBBook(**book.model_dump())
    inserted_book = insert_book(new_db_book, db)

    return inserted_book


@prefix_router.get("/books/search", response_model=list[Book], status_code=status.HTTP_200_OK)
async def search_books(db:Annotated[Session, Depends(get_db)], 
                       title: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       authors: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       lang:Annotated[str|None, Query(min_length=2, max_length=2, examples=["en", "da", "nl"])] = None ):
    
    if not any([title, authors, lang]):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provide at least one filter parameter.")
    
    db_books = select_books_like(title=title, authors=authors, lang=lang, db_sess=db)

    return db_books


@prefix_router.get("/books/{book_id}", response_model=Book, status_code=status.HTTP_200_OK)
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
@prefix_router.get("/books/", response_model=list[Book])
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

# TODO: test this - how is the result paginated
@prefix_router.get("/books/paginated")
async def get_books_paginated(db:Annotated[Session, Depends(get_db)]) -> Page[BookBase]:
    db_books = select_documents_paginated(db)
    books = paginate([BookBase(**b.__dict__) for b in db_books.items])
    return books


@prefix_router.get("/books/documents/", response_model=SearchPage, status_code=status.HTTP_200_OK)
async def get_docs(skip:Annotated[int, Query(description="Number of search result documents to skip", le=100, ge=1)], 
                   take:Annotated[int, Query(description="Number of search result documents to take after skipping", le=100, ge=1)],
                   select:Annotated[list[Literal["book_name", "book_key", "content", "chunk_id", "content_vector", "*"]], Query(description="Fields to select from the vector index")] = ["*"],
                   query:Annotated[str, Query(description="The search query")] = "", ):
    
    sett = get_settings()
    search_client = SearchClient(endpoint=sett.AZURE_SEARCH_ENDPOINT, 
                                 index_name="moby", 
                                 credential=AzureKeyCredential(sett.AZURE_SEARCH_KEY))
    
    select_str = ", ".join(select)
    page = paginated_search(q=query, 
                            search_client=search_client, 
                            skip=skip, 
                            top=take, 
                            select_fields=select_str)

    return page


class GBBookMeta(Book):
    summaries:list[str]
    subjects:list[str]
    languages:list[str]
    authors:list[dict]
    editors:list[dict]
    download_count:int
    copyright:bool
   
    @field_validator("authors", "editors", mode="before")
    def clear_nulls(cls, v:list[dict]):
        d = [{k:v if v else None for k,v in author.items() } for author in v ]
        return d


# TODO: search in Gutendex to show books
@prefix_router.get("/books/gutenberg/{gb_id}", status_code=status.HTTP_200_OK, response_model=GBBookMeta)
async def show_gutenberg_book(gb_id:Annotated[int, Path(description="Gutenberg ID of book", gt=0)]):
    try:
        res = requests.get(f"https://gutendex.com/books/{gb_id}")
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc)
    
    if res.status_code != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=res.json())
    else:
        body = res.json()
        return GBBookMeta(**body)


@prefix_router.get("/books/gutenberg/paginated/", status_code=status.HTTP_200_OK, response_model=list[GBBookMeta])
async def show_gutenberg_books_paginated(page_number:Annotated[int, Query(description="Page number to read from", ge=0)],
                                         number_of_books:Annotated[int, Query(description="Number of books to show", ge=1, le=32)]=32
                                        ):
    try:
        res = requests.get(f"https://gutendex.com/books?page={page_number}")
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc)

    if res.status_code != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=res.json())
    else:
        body = res.json()["results"]
        print(body)
        gb_books = [GBBookMeta(**book_dict) for book_dict in body]
        return gb_books[:number_of_books]


# TODO: post book to vector db by using Gutendex ID
# TODO: have default call to initialize db with e.g. 50 books (and use Celery for long time async job)


app.include_router(prefix_router)
add_pagination(app)

# TODO: list all docs from a book, and paginate the results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)