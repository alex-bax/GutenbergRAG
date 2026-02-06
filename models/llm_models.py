from pydantic import BaseModel, Field


class RankedChunk(BaseModel):
    score: int = Field(ge=0, le=10)
    score_reason: str = Field(..., description="The reasoning for choosing the score")
    uuid_str: str


class RankedChunks(BaseModel):
    ranked_chunks: list[RankedChunk]


class ChunkCitation(BaseModel):
    book_name: str
    chunk_content: str
    chunk_nr: int


class AnswerChunk(BaseModel):
    answer: str
    used_chunks: list[ChunkCitation]
