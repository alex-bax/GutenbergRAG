from db.database import Base
from db.generic_operations import delete_by_field_db, insert_row_db, select_by_pk, insert_if_missing_db, select_where_db
from models.schema import DBBookMetaData,DBBookChunkStats
from sqlalchemy import select, delete, and_, or_
# from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from typing import Any, Type, TypeVar
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page
from sqlalchemy.orm import DeclarativeBase
class BookNotFoundException(Exception):
    pass

async def insert_book_db(
                *,
                book: DBBookMetaData,
                db_sess: AsyncSession,
            ) -> DBBookMetaData:
        book = await insert_row_db(obj=book, db_sess=db_sess)
        return book



async def insert_missing_book_db(book_meta:DBBookMetaData, db_sess:AsyncSession) -> tuple[bool, str]:
    is_inserted, mess = await insert_if_missing_db(
                                obj=book_meta,
                                model=DBBookMetaData,
                                pk_name="gb_id",
                                db_sess=db_sess,
                            )
    return is_inserted, mess

async def insert_missing_chunk_stats_db(chunk_stats:DBBookChunkStats, db_sess:AsyncSession) -> tuple[bool, str]:
    is_inserted, msg = await insert_if_missing_db(
                        obj=chunk_stats,
                        model=DBBookChunkStats,
                        pk_name="id",
                        db_sess=db_sess,
                    )
    return is_inserted, msg


async def select_all_books_db(db_sess:AsyncSession,
                            ) -> list[DBBookMetaData]:
    return await select_where_db(
                    model=DBBookMetaData,
                    conditions=[],
                    db_sess=db_sess,
                )


async def select_books_by_id_db(db_sess:AsyncSession,
                                gb_ids:set[int]=set(),
                            ) -> list[DBBookMetaData]:
    
    conditions = [DBBookMetaData.gb_id.in_(gb_ids)]
    return await select_where_db(
                        model=DBBookMetaData,
                        conditions=conditions,
                        db_sess=db_sess,
                    )
    

async def select_chunk_stats_db_by_id_db(db_sess:AsyncSession,
                                chunk_ids:set[int]=set(),
                            ) -> list[DBBookChunkStats]:
    
    conditions = [DBBookChunkStats.id.in_(chunk_ids)]
    return await select_where_db(
                        model=DBBookChunkStats,
                        conditions=conditions,
                        db_sess=db_sess,
                    )
    

    # obj = await select_by_pk(model=DBBookMetaData, pk_name="gb_id", v)

# async def select_books_db_by_id(book_ids:set[int]|None, 
#                                 db_sess:AsyncSession,
#                                 table_type: Base,
#                                 gb_ids:set[int]|None=None) -> list[DBBookMetaData]:
#     if book_ids is not None and gb_ids is not None:
#         raise ValueError("Provide either book_ids or gb_ids")

#     if gb_ids is not None:
#         stmt = select(DBBookMetaData).filter(DBBookMetaData.gb_id.in_(gb_ids))#.where(DBBookMetaData.gb_id == gb_id)
#     else:
#         assert book_ids is not None
#         stmt = select(DBBookMetaData).filter(DBBookMetaData.id.in_(book_ids))


#     res = await db_sess.execute(stmt)
#     books = res.scalars().all() 
    
#     if not books:
#         books = []
    
#     return list(books)

#TODO refactor these to use GENERIC T also

async def select_books_like_db(title:str|None, 
                            authors:str|None, 
                            db_sess:AsyncSession) -> list[DBBookMetaData]:
    conditions = []         # bool expressions to be joined together
    if title:
        conditions.append(DBBookMetaData.title.ilike(f"%{title}%"))     # case insensitive
    if authors:     # authors can be separated by ;
        conditions.append(or_(*[DBBookMetaData.authors.ilike(f"%{a}%") for a in authors.split(";")]))
    
    return await select_where_db(
                    model=DBBookMetaData,
                    conditions=conditions,
                    db_sess=db_sess,
                )


async def delete_book_db(db_sess:AsyncSession, gb_id:int) -> None:
    await delete_by_field_db(model=DBBookMetaData, field_name="gb_id", value=gb_id, db_sess=db_sess)


# TODO
async def select_documents_paginated_db(db_sess:AsyncSession) -> Page[DBBookMetaData]:
    return await paginate(db_sess, select(DBBookMetaData))

