from db.schema import DBBook
from models.vector_db import ContentChunk
from pydantic import BaseModel, Field, field_validator

from preprocess_book import make_slug_book_key
from search_handler import SearchPage

class BookBase(BaseModel):
    id: int
    title: str

class BookMetaDataResponse(BookBase):
    lang:str = Field(..., description="ISO language code the book is written in", examples=["en", "nl", "da"])
    book_id: int
    authors: str 
    model_config = {"from_attributes": True}        

class Book(BookBase):
    id:int
    book_name:str
    book_key: str
    chunks: list[ContentChunk]       

class ApiResponse(BaseModel):
    data: BookMetaDataResponse|list[BookMetaDataResponse]|SearchPage|None
    job_id: int|None = Field(None, description="Id for async long running jobs when uploading many books")   
    message: str|None    

class GBBookMeta(BaseModel):
    title:str
    id:int = Field(..., title="Gutenberg ID", description="Gutenberg book ID")
    summaries:list[str]
    subjects:list[str]
    languages:list[str]
    authors:list[dict]
    editors:list[dict]
    download_count:int
    formats:dict[str, str]
    copyright:bool

    # Some dicts have null value
    @field_validator("authors", "editors", mode="before")
    def clear_nulls(cls, v:list[dict]):
        d = [{k:v if v else None for k,v in author.items() } for author in v ]
        return d


    def authors_as_str(self) -> str:
        author_names = [a.get("name", "Unknown") for a in self.authors]
        return "; ".join(author_names)

    def _convert_to_db_model(self) -> DBBook:
        authors_str = self.authors_as_str()
        
        return DBBook(
            id=self.id,
            title=self.title,
            authors=authors_str,
            slug_key=make_slug_book_key(title=self.title, gutenberg_id=self.id, author=authors_str),
            lang=self.languages[0],
        )
    
    def get_txt_url(self) -> str|None:
        found = None
        for format, url in self.formats.items():
            if "text/plain;" in format:
                return url
    
        return found