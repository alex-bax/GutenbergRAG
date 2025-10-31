from database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, text

# table=True means this class represent DB table
class Book(Base):
    __tablename__ = "metaDataBook"

    id = Column(Integer,primary_key=True,nullable=False)
    title = Column(String,nullable=False)
    authors = Column(String, nullable=False)        # TODO: add FK to author
    slug_key = Column(String, nullable=False)
    lang = Column(String, nullable=False)
    # content = Column(String,nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))

# TODO: add inheritance, refactor class 
