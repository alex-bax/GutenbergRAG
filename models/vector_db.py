from pydantic import BaseModel, Field, model_validator
from typing import Literal
from constants import EmbeddingDimension

import numpy as np

class EmbeddingVec(BaseModel):
    vector: list[float] = Field(..., description="The embedding vector")
    dim: Literal[EmbeddingDimension.SMALL, EmbeddingDimension.LARGE] = Field(..., description="Dimension of the embedding vector")

    @model_validator(mode="after")      # Run after field validators
    def validate_vector_dimension(self):
        expected_dim = self.dim.value        

        if len(self.vector) != expected_dim:
            raise ValueError(f"Vector dimension {len(self.vector)} does not match specified dimension {expected_dim}")
        
        return self

class ContentChunk(BaseModel):
    uuid_str:str = Field(...)
    book_name:str = Field(..., description="Name/title of the book")
    # book_key:str = Field(..., description="Unique, slug-style key that identifies the book", examples=["frankenstein-or-the-modern-prometheus_shelley-mary-wollstonecraft_84_en", 
    book_id:int = Field(...)
    chunk_id:int = Field(...)
    # chapter_title:str = Field(..., description="Title of the chapter")
    content:str = Field(..., description="Content of the chapter")
    content_vector:EmbeddingVec

    def to_dict(self) -> dict[str, int|str|list[float]]:
        embed_vec = self.content_vector.vector
        return self.__dict__ | { "content_vector": embed_vec }



if __name__ == "__main__":
    # ev = EmbeddingVec(vector=np.arange(0, EmbeddingDimension.SMALL, 0.2).tolist(), dim=EmbeddingDimension.SMALL)
    # print(ev)

    ev_bad = EmbeddingVec(vector=list(range(0, 3, 1)), dim=EmbeddingDimension.SMALL)
    print(ev_bad)



