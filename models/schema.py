from db.database import Base
from sqlalchemy import ARRAY, JSON, Column, Integer, String, Float, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship

# table=True means this class represent a DB table
class DBBookMetaData(Base):
    __tablename__ = "book_metadata"

    gb_id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)#mapped_column(Integer,nullable=False)
    title:Mapped[str] = mapped_column(String,nullable=False)
    authors:Mapped[str] = mapped_column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    summary:Mapped[str] = mapped_column(String, nullable=False)

    # one-to-one relationship
    chunk_stats: Mapped["DBBookChunkStats"] = relationship(
                                                    back_populates="book_metadata",
                                                    uselist=False,
                                                    cascade="all, delete-orphan",
                                                )

class DBBookChunkStats(Base):
    __tablename__ = "book_chunk_stats"
    id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)
    config_id_used:Mapped[int] = mapped_column(Integer, nullable=False)
    title:Mapped[str] = mapped_column(String,nullable=False)

    char_count:Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count:Mapped[int] = mapped_column(Integer, nullable=False)
    token_mean:Mapped[float] = mapped_column(Float, nullable=False)
    token_median:Mapped[float]= mapped_column(Float, nullable=False)
    token_min:Mapped[int]= mapped_column(Integer, nullable=False)
    token_max:Mapped[int]= mapped_column(Integer, nullable=False)
    token_std:Mapped[float]= mapped_column(Float, nullable=False)

    # token_counts: Mapped[list[int]] = mapped_column(
    #                                         ARRAY(Integer),
    #                                         nullable=False,
    #                                     )

    token_counts: Mapped[list[int]] = mapped_column(
                                        ARRAY(Integer).with_variant(JSON, "sqlite"),
                                        nullable=False,
                                    )

    book_meta_fk: Mapped[int] = mapped_column(
                                    ForeignKey("book_metadata.gb_id", ondelete="CASCADE"),
                                    nullable=False,
                                    unique=True,  # enforces 1–1 at the DB level
                                )

    book_metadata: Mapped["DBBookMetaData"] = relationship(
                                                    back_populates="chunk_stats"
                                                )
