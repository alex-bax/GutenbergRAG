import uvicorn, requests
from fastapi import Body, FastAPI, APIRouter, Depends, HTTPException, Query, Path, status
from openai import AzureOpenAI, AsyncAzureOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal
import psycopg2

from azure.search.documents import SearchClient

# from sqlalchemy import select, delete
# from sqlalchemy.orm import Session
from db.database import engine, get_async_db_sess#, SessionLocal
from db.vector_store_abstract import AsyncVectorStore
from sqlalchemy.ext.asyncio import AsyncSession
from db.operations import select_all_books_db, select_books_db, delete_book_db, insert_book_db, select_books_like_db, select_documents_paginated_db, BookNotFoundException

from models.schema import DBBookMetaData
import models.schema as schema
from models.api_response_model import ApiResponse, BookMetaDataResponse, GBBookMeta, QueryResponse
# from models.vector_db import SearchPage
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page, add_pagination, paginate

from converters import gbbookmeta_to_db_obj, db_obj_to_response
from ingestion.book_loader import fetch_book_content_from_id, index_upload_missing_book_ids
from vector_store_utils import  paginated_search
from settings import get_settings, Settings
from retrieval.retrieve import answer_rag

app = FastAPI(title="MobyRAG")
prefix_router = APIRouter(prefix="/v1")

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(schema.Base.metadata.create_all)        # creates the DB tables

# TODO make config obj
async def get_vector_store() -> AsyncVectorStore:
    return await get_settings().get_vector_store()


def get_async_emb_client() -> AsyncAzureOpenAI:
    return get_settings().get_async_emb_client()


@prefix_router.post("/books/", status_code=status.HTTP_201_CREATED)
async def create_book(book:BookMetaDataResponse, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    new_db_book = DBBookMetaData(**book.model_dump())
    await insert_book_db(new_db_book, db)

@prefix_router.get("/books/search", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def search_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)], 
                       title: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       authors: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       lang:Annotated[str|None, Query(min_length=2, max_length=2, examples=["en", "da", "nl"])] = None ):
    
    if not any([title, authors, lang]):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provide at least one filter parameter.")
    
    db_books = await select_books_like_db(title=title, authors=authors, lang=lang, db_sess=db)
    book_metas = [b.to_book_meta_response() for b in db_books]

    return ApiResponse(data=book_metas)


@prefix_router.get("/books/{book_id}", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def get_book(book_id:int, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    db_books = None
    try:
        db_books = await select_books_db(set([book_id]), db)
    except BookNotFoundException: 
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")

    if not db_books:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Book with id {book_id} empty")

    book_meta_objs = [db_obj_to_response(b) for b in db_books]
    return ApiResponse(data=book_meta_objs)

# TODO: could be slow if DB is huge, use pagination instead
@prefix_router.get("/books/", response_model=ApiResponse)
async def get_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    books = await select_all_books_db(db)
    return ApiResponse(data=[b.to_book_meta_response() for b in books])


@prefix_router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_book(book_id:int, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    try:
        await delete_book_db(book_id, db)
    except BookNotFoundException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'{exc}')


# TODO: test this - how is the result paginated
@prefix_router.get("/books/paginated")
async def get_books_paginated(db:Annotated[AsyncSession, Depends(get_async_db_sess)]) -> Page[BookMetaDataResponse]:
    db_books = await select_documents_paginated_db(db)
    books = paginate([BookMetaDataResponse(**b.__dict__) for b in db_books.items])
    return books


@prefix_router.get("/index/documents/", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def search_docs_by_texts(skip:Annotated[int, Query(description="Number of search result documents to skip", le=100, ge=1)], 
                                take:Annotated[int, Query(description="Number of search result documents to take after skipping", le=100, ge=1)],
                                settings:Annotated[Settings, Depends(get_settings)],
                                # select:Annotated[list[Literal["book_name", "book_id", "content", "chunk_id", "content_vector", "*"]], Query(description="Fields to select from the vector index")] = ["*"],
                                query:Annotated[str, Query(description="The search query")] = "", 
                                ):
    assert settings._vector_store
    page = await settings._vector_store.paginated_search_by_text(text_query=query, 
                                                                skip=skip, 
                                                                limit=take, 
                                                            ) 
    return ApiResponse(data=page)

#TODO: post book to vector db by using Gutendex ID
# no body needed, only gutenberg id since we're uploading from Gutenberg 
@prefix_router.post("/index", status_code=status.HTTP_201_CREATED, response_model=ApiResponse)
async def upload_book_to_index(gutenberg_ids:Annotated[set[int], Body(description="Gutenberg IDs to upload", min_length=1, max_length=50)],
                      db:Annotated[AsyncSession, Depends(get_async_db_sess)],
                      settings:Annotated[Settings, Depends(get_settings)]
                    ):
    info = ""
    book_added = None
    # create_missing_search_index(search_index_client=settings.get_index_client())
    
    books_uploaded = await index_upload_missing_book_ids(book_ids=gutenberg_ids, sett=settings)
    resp_book_uploaded = None

    if len(books_uploaded) > 0:
        for b in books_uploaded:
            await insert_book_db(book=gbbookmeta_to_db_obj(b), db_sess=db)
        resp_book_uploaded = books_uploaded
    else:
        info = f"Book ids:{gutenberg_ids} already in index {settings.INDEX_NAME}"

        try:
            books_from_db = await select_books_db(book_ids=None, db_sess=db, gb_ids=gutenberg_ids)
            resp_book_uploaded = books_from_db
        except BookNotFoundException: 
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ids {gutenberg_ids} not found in DB, but was in index")

    book_meta_objs = [db_obj_to_response(b) for b in resp_book_uploaded]
    return ApiResponse(data=book_meta_objs, message=info) 


#TODO: look up specific chunk by uuid?
@prefix_router.delete("/index/{gutenberg_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book_from_index(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to delete", gt=0)],
                                # search_client:Annotated[SearchClient, Depends(get_vector_store)],
                                settings:Annotated[Settings, Depends(get_settings)],
                                db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    
    vec_store = await settings.get_vector_store()
    missing_ids = await vec_store.get_missing_ids(book_ids=set([gutenberg_id]))      # get book id if missing
    
    err_mess_not_found = ""
    if not missing_ids or len(missing_ids) > 0:
        err_mess_not_found =f"No items in vector found with book_id {gutenberg_id}"
    else:
        await vec_store.delete_books(book_ids=missing_ids)

    try:
        await delete_book_db(book_id=None, gb_id=gutenberg_id, db_sess=db)
    except BookNotFoundException:
        err_mess_not_found += f"\nBook with id {gutenberg_id} not found in DB"
    
    if len(err_mess_not_found) > 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_mess_not_found)

    

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
                        settings:Annotated[Settings, Depends(get_settings)],
                        top_n_matches:Annotated[int, Query(description="Number of matching chunks to include in response", gt=0, lt=50)]=7,
                        only_gb_book_id:Annotated[int|None, Query(description="Filter out all other books than this", gt=0)] = None):

    llm_resp = await answer_rag(query=query, 
                                sett=settings,
                                top_n_matches=top_n_matches)

    return ApiResponse(data=llm_resp)


app.include_router(prefix_router)
add_pagination(app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)