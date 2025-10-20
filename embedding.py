from pydantic import BaseModel, Field, model_validator, ValidationError, ValidationInfo
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


if __name__ == "__main__":
    ev = EmbeddingVec(vector=np.arange(0, EmbeddingDimension.SMALL, 0.2).tolist(), dim=EmbeddingDimension.SMALL)
    print(ev)

    ev_bad = EmbeddingVec(vector=list(range(0, 3, 1)), dim=EmbeddingDimension.SMALL)
    print(ev_bad)