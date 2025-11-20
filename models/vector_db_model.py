from pydantic import BaseModel, Field, model_validator
from typing import Literal
from constants import EmbeddingDimension

class EmbeddingVec(BaseModel):
    vector: list[float] = Field(..., description="The embedding vector")
    dim: Literal[EmbeddingDimension.SMALL, EmbeddingDimension.LARGE] = Field(..., description="Dimension of the embedding vector")

    @model_validator(mode="after")      # Run after field validators
    def validate_vector_dimension(self):
        expected_dim = self.dim.value        

        if len(self.vector) != expected_dim:
            raise ValueError(f"Vector dimension {len(self.vector)} does not match specified dimension {expected_dim}")
        
        return self

class ContentUploadChunk(BaseModel):
    uuid_str:str = Field(...)
    book_name:str = Field(..., description="Name/title of the book")
    book_id:int = Field(...)
    chunk_nr:int = Field(..., description="Nth chunk of all chunks, e.g. if 6, then it's the 6th chunk")
    content:str = Field(..., description="Content of the chapter")
    content_vector:EmbeddingVec

    def to_dict(self) -> dict[str, int|str|list[float]]:
        embed_vec = self.content_vector.vector
        return self.__dict__ | { "content_vector": embed_vec }


# Uses optional since user search request can toggle fields on/off
class SearchChunk(BaseModel):
    uuid_str: str|None
    chunk_nr: int|None = Field(None,  description="Nth chunk of all chunks")
    book_name: str|None
    book_id: int|None
    content: str|None
    search_score: float

class SearchPage(BaseModel):
    items: list[SearchChunk]
    skip_n: int = Field(..., title="Skip N Items", description="Number of items from search result to skip")
    top: int = Field(..., title="Top", description="Starting from the 'skip', take the next 'top' no. items from the search result")
    total_count: int | None = Field(None, description="Total count of results found from the query")




if __name__ == "__main__":
    # ev = EmbeddingVec(vector=np.arange(0, EmbeddingDimension.SMALL, 0.2).tolist(), dim=EmbeddingDimension.SMALL)
    # print(ev)

    ev_bad = EmbeddingVec(vector=list(range(0, 3, 1)), dim=EmbeddingDimension.SMALL)
    print(ev_bad)



