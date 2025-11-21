from __future__ import annotations
from models.vector_db_model import SearchPage, SearchChunk
from pydantic import BaseModel, Field, field_validator


class QueryResponse(BaseModel):
    answer:str
    citations:list[SearchChunk] #?
    #canditates:list[]  #TODO: add content chunks?

class BookBase(BaseModel):
    id: int
    gb_id: int = Field(..., title="Gutenberg book ID")
    title: str

class BookMetaDataResponse(BookBase):
    lang:str = Field(..., description="ISO language code the book is written in", examples=["en", "nl", "da"])
    authors: str 
    model_config = {"from_attributes": True}        

# class Book(BookBase):
#     book_name:str
#     chunks: list[VectorChunk]       

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
    
    def get_txt_url(self) -> str|None:
        found = None
        for format, url in self.formats.items():
            if "text/plain;" in format:
                return url
    
        return found
    

class ApiResponse(BaseModel):
    data: BookMetaDataResponse|list[BookMetaDataResponse]|SearchPage|GBBookMeta|list[GBBookMeta]|QueryResponse|None = Field(default=None)
    job_id: int|None = Field(default=None, description="Id for long running async jobs when uploading many books to index at once")   
    message: str|None = None    


