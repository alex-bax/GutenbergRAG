from db.schema import DBBook
from sqlalchemy import select, delete, and_, or_
from sqlalchemy.orm import Session

from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page

class BookNotFoundException(Exception):
    pass

def select_all_books(db_sess:Session) -> list[DBBook]:
    stmt = select(DBBook)
    res = db_sess.execute(stmt)
    book_rows = list(res.scalars().all())       # [(<schema.Book object at 0x0000019D638986E0>,)]

    return book_rows

def select_book(book_id:int, db_sess:Session) -> DBBook:
    stmt = select(DBBook).where(DBBook.id == book_id)
    res = db_sess.execute(stmt)
    book = res.scalars().one_or_none()
    
    if not book:
        raise BookNotFoundException(f"Book with id {book_id} not found")
    
    return book

def select_books_like(title:str|None, authors:str|None, lang:str|None, db_sess:Session) -> list[DBBook]:
    conditions = []         # bool expressions to be joined together
    stmt = select(DBBook)

    if lang:
        conditions.append(DBBook.lang == lang)
    if title:
        conditions.append(DBBook.title.ilike(f"%{title}%"))     # case insensitive
    # authors can be separated by ;
    if authors:
        conditions.append(or_(*[DBBook.authors.ilike(f"%{a}%") for a in authors.split(";")]))
 
    if conditions:
        stmt = stmt.where(and_(*conditions))
    
    res = db_sess.execute(stmt)
    
    return list(res.scalars().all())


def delete_book(book_id:int, db_sess:Session) -> None:
    stmt = delete(DBBook).where(DBBook.id == book_id)
    res = db_sess.execute(stmt)
    db_sess.commit()
    
    if res.rowcount == 0:       # type:ignore
        raise BookNotFoundException(f"Book with id {book_id} not found")


def insert_book(book:DBBook, db_sess:Session) -> None:
    db_sess.add(book)
    db_sess.commit()
    db_sess.refresh(book)


def select_documents_paginated(db_sess:Session) -> Page[DBBook]:
    return paginate(db_sess, select(DBBook))

