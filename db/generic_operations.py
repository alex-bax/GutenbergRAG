from collections.abc import Sequence
from models.schema import DBBookMetaData,DBBookChunkStats
from sqlalchemy import select, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from typing import Any, Type, TypeVar
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.elements import ColumnElement

T = TypeVar("T", bound=DeclarativeBase)      

async def insert_row_db(
                *,
                obj: T,
                db_sess: AsyncSession,
            ) -> T:
    """
    Generic insert helper.
    - Adds object
    - Commits transaction
    - Refreshes object so PKs are populated
    """
    db_sess.add(obj)
    await db_sess.commit()
    await db_sess.refresh(obj)
    return obj

async def insert_if_missing_db(
                *,
                obj: T,
                model: Type[T],
                pk_name: str,
                db_sess: AsyncSession,) -> tuple[bool, str]:
    """
    Returns: (inserted, message)
    """
    pk_value = getattr(obj, pk_name)

    existing = await select_by_pk(
                            model=model, pk_name=pk_name, pk_value=pk_value, db_sess=db_sess
                        )
    if existing is not None:
        return False, f"Already in DB: {model.__name__}({pk_name}={pk_value})"

    obj_added = await insert_row_db(obj=obj, db_sess=db_sess)
    # await db_sess.flush()       # flush so DB-generated PKs get populated if relevant
    return True, f"Inserted: {model.__name__}({pk_name}={getattr(obj_added, pk_name)})"



async def select_by_pk(
                *,
                model: Type[T],
                pk_name: str,
                pk_value: Any,
                db_sess: AsyncSession) -> T | None:
    pk_col = getattr(model, pk_name)
    res = await db_sess.execute(select(model).where(pk_col == pk_value))
    return res.scalar_one_or_none()



async def select_where_db(
    *,
    model: Type[T],
    conditions: Sequence[ColumnElement[bool]] | None,
    db_sess: AsyncSession,
) -> list[T]:
    stmt = select(model)
    if conditions:
        stmt = stmt.where(*conditions)  # AND by default across multiple where() args
    res = await db_sess.execute(stmt)
    return list(res.scalars().all())


async def delete_by_field_db(
    *,
    model: Type[T],
    field_name: str,
    value: Any,
    db_sess: AsyncSession,
) -> int:
    """
    Deletes rows where model.<field_name> == value.
    Returns number of rows deleted.
    """
    col = getattr(model, field_name)
    stmt = delete(model).where(col == value)
    res = await db_sess.execute(stmt)
    await db_sess.commit()
    return int(res.rowcount or 0)       # type:ignore