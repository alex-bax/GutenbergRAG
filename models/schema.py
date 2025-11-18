from db.database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column

# table=True means this class represent a DB table
class DBBookMetaData(Base):
    __tablename__ = "metaDataBook"

    id:Mapped[int] = mapped_column(Integer,primary_key=True,nullable=False, autoincrement=True)
    gb_id:Mapped[int] = mapped_column(Integer,nullable=False)
    title:Mapped[str] = mapped_column(String,nullable=False)
    authors:Mapped[str] = mapped_column(String, nullable=False)
    lang:Mapped[str] = mapped_column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


