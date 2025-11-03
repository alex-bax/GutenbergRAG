from db.schema import DBBook
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

class BookNotFoundException(Exception):
    pass

def select_all_books(db_sess:Session) -> list[DBBook]:
    stmt = select(DBBook)
    res = db_sess.execute(stmt)
    book_rows = list(res.scalars().all())       # [(<schema.Book object at 0x0000019D638986E0>,)]
    # books = [b[0] for b in book_rows]
    return book_rows

def select_book(book_id:int, db_sess:Session) -> DBBook:
    stmt = select(DBBook).where(DBBook.id == book_id)
    res = db_sess.execute(stmt)
    book = res.scalars().one_or_none()
    
    if not book:
        raise BookNotFoundException(f"Book with id {book_id} not found")
    
    return book

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

