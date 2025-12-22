from db.database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column

# table=True means this class represent a DB table
class DBBookMetaData(Base):
    __tablename__ = "book_metadata"

    # id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)
    gb_id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)#mapped_column(Integer,nullable=False)
    title:Mapped[str] = mapped_column(String,nullable=False)
    authors:Mapped[str] = mapped_column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    summary:Mapped[str] = mapped_column(String, nullable=False)


class DBChunkStats(Base):
    __tablename__ = "book_chunk_stats"
    id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)

    char_count:Mapped[int]
    chunk_count:Mapped[int]
    token_mean:Mapped[float]
    token_median:Mapped[float]
    token_min:Mapped[int]
    token_max:Mapped[int]
    token_std:Mapped[float]
