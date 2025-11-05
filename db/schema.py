from models.api_response import BookMetaDataResponse
from db.database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.sql import func

# table=True means this class represent a DB table
class DBBookMetaData(Base):
    __tablename__ = "metaDataBook"

    id = Column(Integer,primary_key=True,nullable=False, autoincrement=True)
    gb_id = Column(Integer,nullable=False)
    title = Column(String,nullable=False)
    authors = Column(String, nullable=False)        
    lang = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    def to_book_meta_response(self) -> BookMetaDataResponse:
        return BookMetaDataResponse(     
            id=self.id,             # type:ignore                 
            gb_id=self.id,          # type:ignore
            title=self.title,       # type:ignore
            authors=self.authors,   # type:ignore
            lang=self.lang,         # type:ignore
        )

