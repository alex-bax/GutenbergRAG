from contextlib import asynccontextmanager
import uvicorn, requests
from fastapi import Body, FastAPI, APIRouter, Depends, HTTPException, Query, Path, status
from openai import AsyncAzureOpenAI
from typing import Annotated
import psycopg2

from db.database import engine, get_async_db_sess, Base
from db.vector_store_abstract import AsyncVectorStore
from sqlalchemy.ext.asyncio import AsyncSession
from db.operations import select_all_books_db, select_books_db_by_id, delete_book_db,  select_books_like_db, select_documents_paginated_db, BookNotFoundException


from models.api_response_model import ApiResponse, BookMetaDataResponse, BookMetaApiResponse, GBBookMeta, GBMetaApiResponse, QueryResponseApiResponse, SearchApiResponse
# from models.api_response_model import SearchPage
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page, add_pagination, paginate

from converters import gbbookmeta_to_db_obj, db_obj_to_response
from ingestion.book_loader import fetch_book_content_from_id, upload_missing_book_ids
from settings import get_settings, Settings
from retrieval.retrieve import answer_rag


# async def init_models():
#     async with engine.begin() as conn:
#         await conn.run_sync(schema.Base.metadata.create_all)        # creates the DB tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Running startup, creating tables...")
    async with engine.begin() as conn:      # startup
        await conn.run_sync(Base.metadata.create_all)

    yield  # Now the app starts serving

    await engine.dispose()  # shutdown


app = FastAPI(title="MobyRAG", lifespan=lifespan)
prefix_router = APIRouter(prefix="/v1")

# TODO make config obj
async def get_vector_store() -> AsyncVectorStore:
    return await get_settings().get_vector_store()


def get_async_emb_client() -> AsyncAzureOpenAI:
    return get_settings().get_async_emb_client()


@prefix_router.get("/books/search", response_model=BookMetaApiResponse, status_code=status.HTTP_200_OK)
async def search_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)], 
                       title: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       authors: Annotated[str|None, Query(min_length=3, max_length=100)] = None, 
                       lang:Annotated[str|None, Query(min_length=2, max_length=2, examples=["en", "da", "nl"])] = None ):
    
    if not any([title, authors, lang]):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Provide at least one filter parameter.")
    
    db_books = await select_books_like_db(title=title, authors=authors, lang=lang, db_sess=db)
    book_meta_objs = [db_obj_to_response(b) for b in db_books]

    return BookMetaApiResponse(data=book_meta_objs)


@prefix_router.get("/books/{book_id}", response_model=BookMetaApiResponse, status_code=status.HTTP_200_OK)
async def get_book(book_id:int, db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    db_books = None
    
    db_books = await select_books_db_by_id(set([book_id]), db)

    if len(db_books) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with id {book_id} not found")

    if not db_books:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Book with id {book_id} empty")

    book_meta_objs = [db_obj_to_response(b) for b in db_books]
    return BookMetaApiResponse(data=book_meta_objs)


@prefix_router.get("/books/", response_model=BookMetaApiResponse)
async def get_books(db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    books = await select_all_books_db(db)
    return BookMetaApiResponse(data=[db_obj_to_response(b) for b in books])


# TODO: test this - how is the result paginated
@prefix_router.get("/books/paginated")
async def get_books_paginated(db:Annotated[AsyncSession, Depends(get_async_db_sess)]) -> Page[BookMetaDataResponse]:
    db_books = await select_documents_paginated_db(db)
    books = paginate([BookMetaDataResponse(**b.__dict__) for b in db_books.items])
    return books


@prefix_router.get("/index/{gutenberg_id}", response_model=SearchApiResponse, status_code=status.HTTP_200_OK)
async def get_book_from_index(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to delete", gt=0)],
                                settings:Annotated[Settings, Depends(get_settings)],
                                db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    vec_store = await settings.get_vector_store()
    book_search_page = await vec_store.get_paginated_chunks_by_book_ids(book_ids=set([gutenberg_id]))

    return SearchApiResponse(data=[book_search_page])


@prefix_router.get("/index/{gutenberg_id}", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def get_chunk_count_in_book(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to delete", gt=0)]):
    ...

@prefix_router.get("/index/documents/", response_model=SearchApiResponse, status_code=status.HTTP_200_OK)
async def search_index_by_texts(skip:Annotated[int, Query(description="Number of search result documents to skip", le=100, ge=0)], 
                                take:Annotated[int, Query(description="Number of search result documents to take after skipping", le=100, ge=1)],
                                settings:Annotated[Settings, Depends(get_settings)],
                                # select:Annotated[list[Literal["book_name", "book_id", "content", "chunk_id", "content_vector", "*"]], Query(description="Fields to select from the vector index")] = ["*"],
                                query:Annotated[str, Query(description="The search query")] = "", 
                                ):
    vec_store = await settings.get_vector_store()
    page = await vec_store.paginated_search_by_text(text_query=query, 
                                                        skip=skip, 
                                                        limit=take, 
                                                    ) 
    return SearchApiResponse(data=[page])

#TODO: post book to vector db by using Gutendex ID
# no body needed, only gutenberg id since we're uploading from Gutenberg 
@prefix_router.post("/index", status_code=status.HTTP_201_CREATED, response_model=GBMetaApiResponse)
async def upload_book_to_index(gutenberg_ids:Annotated[list[int], Body(description="Unique Gutenberg IDs to upload", min_length=1, max_length=30)],
                                db_sess:Annotated[AsyncSession, Depends(get_async_db_sess)],
                                settings:Annotated[Settings, Depends(get_settings)]):
    info = ""

    if len(gutenberg_ids) != len(set(gutenberg_ids)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="gutenberg_ids must be unique",
        )

    gb_books_uploaded, info = await upload_missing_book_ids(book_ids=set(gutenberg_ids), sett=settings, db_sess=db_sess)

    if len(gb_books_uploaded) == 0:
        info += f"\nBook ids:{gutenberg_ids} already in index {settings.active_collection}"

    return GBMetaApiResponse(data=gb_books_uploaded, message=info) 



#TODO: add delete and lookup by specific chunk by uuid?
@prefix_router.delete("/index/{gutenberg_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book_from_index(gutenberg_id:Annotated[int, Path(description="Gutenberg ID to delete", gt=0)],
                                settings:Annotated[Settings, Depends(get_settings)],
                                db:Annotated[AsyncSession, Depends(get_async_db_sess)]):
    
    vec_store = await settings.get_vector_store()
    missing_ids = await vec_store.get_missing_ids_in_store(book_ids=set([gutenberg_id]))      # get book id if missing
    
    err_mess_not_found = ""
    if not missing_ids or len(missing_ids) == 0:
        await vec_store.delete_books(book_ids=set([gutenberg_id]))
    else:
        err_mess_not_found =f"No items in vector found with book_id {gutenberg_id}"

    try:
        await delete_book_db(book_id=None, gb_id=gutenberg_id, db_sess=db)
    except BookNotFoundException:
        err_mess_not_found += f"\nBook with id {gutenberg_id} not found in DB"
    
    if len(err_mess_not_found) > 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_mess_not_found)

    

@prefix_router.get("/books/gutenberg/{gutenberg_id}", status_code=status.HTTP_200_OK, response_model=GBMetaApiResponse)
async def show_gutenberg_book(gutenberg_id:Annotated[int, Path(description="Gutenberg ID of book", gt=0)]):
    try:
        res = requests.get(f"https://gutendex.com/books/{gutenberg_id}")
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc)
    
    if res.status_code != status.HTTP_200_OK:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=res.json())
    else:
        body = res.json()
        return GBMetaApiResponse(data=[GBBookMeta(**body)])


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

@prefix_router.get("/query/", status_code=status.HTTP_202_ACCEPTED, response_model=QueryResponseApiResponse)
async def answer_query(query:Annotated[str, Query()],
                        settings:Annotated[Settings, Depends(get_settings)],
                        top_n_matches:Annotated[int, Query(description="Number of matching chunks to include in response", gt=0, lt=50)]=7,
                        only_gb_book_id:Annotated[int|None, Query(description="Filter out all other books than this", gt=0)] = None):

    llm_resp = await answer_rag(query=query, 
                                sett=settings,
                                top_n_matches=top_n_matches)

    return QueryResponseApiResponse(data=llm_resp)




app.include_router(prefix_router)
add_pagination(app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)