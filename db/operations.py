from models.schema import DBBookMetaData
from sqlalchemy import select, delete, and_, or_
# from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page

class BookNotFoundException(Exception):
    pass

# def insert_book(book:DBBookMetaData, db_sess:Session) -> None:
async def insert_book_db(book:DBBookMetaData, db_sess:AsyncSession) -> None:
    db_sess.add(book)
    await db_sess.commit()
    await  db_sess.refresh(book)

async def insert_missing_book_db(book:DBBookMetaData, db_sess:AsyncSession) -> str:
    existing_books = await select_books_db_by_id(set([book.id]), db_sess=db_sess)
    
    if len(existing_books) == 0:
        await insert_book_db(book=book, db_sess=db_sess)
        return f"\n Didn't find book {book.title} {book.id} in metadata DB - inserting it now"
    else:
        return f"\n Already in metadata DB: {book.title} {book.id}"


async def select_all_books_db(db_sess:AsyncSession) -> list[DBBookMetaData]:
    stmt = select(DBBookMetaData)
    res =await db_sess.execute(stmt)
    book_rows = list(res.scalars().all())       # [(<schema.Book object at 0x0000019D638986E0>,)]

    return book_rows

async def select_books_db_by_id(book_ids:set[int]|None, db_sess:AsyncSession, gb_ids:set[int]|None=None) -> list[DBBookMetaData]:
    if gb_ids:
        stmt = select(DBBookMetaData).filter(DBBookMetaData.gb_id.in_(gb_ids))#.where(DBBookMetaData.gb_id == gb_id)
    else:
        stmt = select(DBBookMetaData).where(DBBookMetaData.id == book_ids)

    res = await db_sess.execute(stmt)
    books = res.scalars().all() 
    
    if not books:
        books = []
    
    return list(books)

async def select_books_like_db(title:str|None, authors:str|None, lang:str|None, db_sess:AsyncSession) -> list[DBBookMetaData]:
    conditions = []         # bool expressions to be joined together
    stmt = select(DBBookMetaData)

    if lang:
        conditions.append(DBBookMetaData.lang == lang)
    if title:
        conditions.append(DBBookMetaData.title.ilike(f"%{title}%"))     # case insensitive
    # authors can be separated by ;
    if authors:
        conditions.append(or_(*[DBBookMetaData.authors.ilike(f"%{a}%") for a in authors.split(";")]))
 
    if conditions:
        stmt = stmt.where(and_(*conditions))
    
    res = await db_sess.execute(stmt)
    
    return list(res.scalars().all())


async def delete_book_db(book_id:int|None, db_sess:AsyncSession, gb_id:int|None=None) -> None:
    
    if gb_id:
        stmt = delete(DBBookMetaData).where(DBBookMetaData.gb_id == gb_id)
    else:
        stmt = delete(DBBookMetaData).where(DBBookMetaData.id == book_id)
    
    res = await db_sess.execute(stmt)
    await db_sess.commit()
    
    if res.rowcount == 0:       # type:ignore
        raise BookNotFoundException(f"Book with id {book_id} not found")



async def select_documents_paginated_db(db_sess:AsyncSession) -> Page[DBBookMetaData]:
    return await paginate(db_sess, select(DBBookMetaData))

