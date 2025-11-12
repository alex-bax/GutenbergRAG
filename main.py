from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Path, status
from openai import AzureOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal,AsyncIterator
import psycopg2
import uvicorn, requests

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

# from sqlalchemy import select, delete
# from sqlalchemy.orm import Session
from db.database import engine, get_async_db_sess#, SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from db.operations import select_all_books, select_book, delete_book, insert_book, select_books_like, select_documents_paginated, BookNotFoundException

from models.schema import DBBookMetaData
import models.schema as schema
from moby import _make_limiters
from models.api_response import ApiResponse, BookMetaDataResponse, GBBookMeta, QueryResponse
# from models.vector_db import SearchPage
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page, add_pagination, paginate

from converters import gbbookmeta_to_db
from load_book import fetch_book_content_from_id
from preprocess_book import make_slug_book_key
from search_handler import check_missing_books_in_index, paginated_search, create_missing_search_index, upload_to_index_async
from settings import get_settings
from retrieve import answer_api

app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")

# schema.Base.metadata.create_all(bind=engine)        # creates the DB tables

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(schema.Base.metadata.create_all)


# async def get_db() -> AsyncIterator[AsyncSession]:
#     async with AsyncSessionLocal() as session:
#         yield session


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

def get_search_client() -> SearchClient:
    return get_settings().get_search_client()

def get_index_client() -> SearchIndexClient:
    return get_settings().get_index_client()

def get_llm_client() -> AzureOpenAI:
    return get_settings().get_llm_client()

def get_emb_client() -> AzureOpenAI:
    return get_settings().get_emb_client()


@prefix_router.post("/books/", status_code=status.HTTP_201_CREATED)
async def create_book(book:BookMetaDataResponse, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    new_db_book = DBBookMetaData(**book.model_dump())
    await insert_book(new_db_book, db)

@prefix_router.get("/books/search", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def search_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)], 
                       title: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       authors: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       lang:Annotated[str|None, Query(min_length=2, max_length=2, examples=["en", "da", "nl"])] = None ):
    
    if not any([title, authors, lang]):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provide at least one filter parameter.")
    
    db_books = await select_books_like(title=title, authors=authors, lang=lang, db_sess=db)
    book_metas = [b.to_book_meta_response() for b in db_books]

    return ApiResponse(data=book_metas)


@prefix_router.get("/books/{book_id}", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def get_book(book_id:int, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    book = None
    try:
        book = await select_book(book_id, db)
    except BookNotFoundException: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")

    if not book:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Book with id {book_id} empty")

    return ApiResponse(data=book.to_book_meta_response())

# TODO: could be slow if DB is huge, use pagination instead
@prefix_router.get("/books/", response_model=ApiResponse)
async def get_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    books = await select_all_books(db)
    return ApiResponse(data=[b.to_book_meta_response() for b in books])


@prefix_router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_book(book_id:int, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    try:
        await delete_book(book_id, db)
    except BookNotFoundException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'{exc}')


# TODO: test this - how is the result paginated
@prefix_router.get("/books/paginated")
async def get_books_paginated(db:Annotated[AsyncSession, Depends(get_async_db_sess)]) -> Page[BookMetaDataResponse]:
    db_books = await select_documents_paginated(db)
    books = paginate([BookMetaDataResponse(**b.__dict__) for b in db_books.items])
    return books


@prefix_router.get("/index/documents/", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def get_docs(skip:Annotated[int, Query(description="Number of search result documents to skip", le=100, ge=1)], 
                   take:Annotated[int, Query(description="Number of search result documents to take after skipping", le=100, ge=1)],
                   search_client:Annotated[SearchClient, Depends(get_search_client)],
                   select:Annotated[list[Literal["book_name", "book_id", "content", "chunk_id", "content_vector", "*"]], Query(description="Fields to select from the vector index")] = ["*"],
                   query:Annotated[str, Query(description="The search query")] = "", 
                   ):
    
    select_str = ", ".join(select)
    page = paginated_search(q=query, 
                            search_client=search_client, 
                            skip=skip, 
                            top=take, 
                            select_fields=select_str)

    return ApiResponse(data=page)

#TODO: post book to vector db by using Gutendex ID
#TODO: make it work with list of ids?
# no body needed, only gutenberg id since we're uploading from Gutenberg 
@prefix_router.post("/index/{gutenberg_id}", status_code=status.HTTP_201_CREATED, response_model=ApiResponse)
async def upload_book_to_index(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to upload", gt=0)],
                      search_client:Annotated[SearchClient, Depends(get_search_client)],
                      index_client:Annotated[SearchIndexClient, Depends(get_index_client)],
                      emb_client:Annotated[AzureOpenAI, Depends(get_emb_client)],
                      db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    sett = get_settings()
    #TODO: consider how to manage these search instances smartly in a deployed env
    info = ""

    book_added = None

    create_missing_search_index(search_index_client=index_client)

    book_content, gb_meta = fetch_book_content_from_id(gutenberg_id=gutenberg_id)
    req_limiter, tok_limiter = _make_limiters()

    if  len(check_missing_books_in_index(search_client=search_client, book_ids=[gb_meta.id])) > 0:
        chunks_added = await upload_to_index_async(search_client=search_client, 
                                        embed_client=emb_client,
                                        token_limiter=tok_limiter,
                                        request_limiter=req_limiter,
                                        book_meta=gb_meta,
                                        raw_book_content=book_content
                                    )
        
        await insert_book(book=gbbookmeta_to_db(gb_meta), db_sess=db)
        book_added = gb_meta
    else:
        info = f"Book already in index {sett.INDEX_NAME} as '{gb_meta.title}'. Fetched meta data from DB"

        try:
            book = await select_book(book_id=None, db_sess=db, gb_id=gutenberg_id)
        except BookNotFoundException: 
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {gutenberg_id} not found, but was in index")

        book_added = book.to_book_meta_response()

    return ApiResponse(data=book_added, message=info) 

#TODO: look up specific chunk by uuid?
@prefix_router.delete("/index/{gutenberg_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book_from_index(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to delete", gt=0)],
                                search_client:Annotated[SearchClient, Depends(get_search_client)],
                                db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    # Ensure that book exists before deleting it from index
    try:
        await delete_book(book_id=None, gb_id=gutenberg_id, db_sess=db)
    except BookNotFoundException: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {gutenberg_id} not found in DB")
    
    # get all documents with matching book_key
    result = search_client.search(search_text="*",  # match all docs
                                filter=f"book_id eq {gutenberg_id}",
                                select=["uuid_str"],  # only fetch the ID "PK" field 
                                )
    
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"No index docs found with book_id {gutenberg_id}")
    else:
        search_client.delete_documents(list(result))

  

@prefix_router.get("/books/gutenberg/{gutenberg_id}", status_code=status.HTTP_200_OK, response_model=ApiResponse)
async def show_gutenberg_book(gutenberg_id:Annotated[int, Path(description="Gutenberg ID of book", gt=0)]):
    try:
        res = requests.get(f"https://gutendex.com/books/{gutenberg_id}")
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc)
    
    if res.status_code != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=res.json())
    else:
        body = res.json()
        return ApiResponse(data=GBBookMeta(**body))


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

# TODO: have default call to initialize db with e.g. 50 books (and use Celery for long time async job)
        # e.g. populate index

@prefix_router.get("/query/", status_code=status.HTTP_202_ACCEPTED, response_model=ApiResponse)
async def answer_query(query:Annotated[str, Query()],
                      search_client:Annotated[SearchClient, Depends(get_search_client)],
                      llm_client:Annotated[AzureOpenAI, Depends(get_llm_client)],
                      emb_client:Annotated[AzureOpenAI, Depends(get_emb_client)],
                      top_n_matches:Annotated[int, Query(description="Number of matching chunks to include in response", gt=0, lt=50)]=7,
                      only_gb_book_id:Annotated[int|None, Query(description="Filter out all other books than this", gt=0)] = None):

    sett = get_settings()

    llm_resp = answer_api(query=query, 
                      search_client=search_client, 
                      embed_client=emb_client, 
                      llm_client=llm_client,
                      top_n_matches=top_n_matches,
                      embed_model_deployed=sett.EMBED_MODEL_DEPLOYMENT, 
                      llm_model_deployed=sett.LLM_MODEL_DEPLOYMENT)
    # TODO: add the ans to the reponse type
    return ApiResponse(data=llm_resp)


app.include_router(prefix_router)
add_pagination(app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)