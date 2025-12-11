import json
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pathlib import Path
from typing import Literal

MAX_TOKENS = 8000
CHUNK_SIZE = 400
OVERLAP = 100
TOKEN_PR_MIN = 501_000
REQUESTS_PR_MIN = 3000

MIN_SEARCH_SCORE = 0.01

ID_FRANKENSTEIN = 84
ID_DR_JEK_MR_H = 42
ID_MOBY = 2701
ID_SHERLOCK = 1661

DEF_BOOK_NAMES_TO_IDS = {
    "The Adventures of Sherlock Holmes": ID_SHERLOCK, 
    "The Strange Case of Dr. Jekyll and Mr. Hyde": ID_DR_JEK_MR_H, 
    "The Federalist Papers": 1404, 
    "Moby Dick; Or, The Whale": ID_MOBY,
    "Meditations":2680,
    "The King in Yellow":8492
}
VER_PREFIX = "v1"


class EmbeddingDimension(int, Enum):
    SMALL = 1536
    LARGE = 3072



class ChunkConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 40
    chunk_strategy:Literal["fixed", "semantic"]
    embed_model:str
    embed_dim:EmbeddingDimension
    default_ids_used:dict[str,int]


class RetrievalConfig(BaseModel):
    top_k: int = 8
    vector_db:Literal["Qdrant", "Azure AI Search"]
    collection:str
    vector_db_type:Literal["vector", "hybrid_search"]
    top_k:int

class RerankConfig(BaseModel):
    enabled: bool = True
    batch_size:int
    model:str
    rank_method:str


class GenerationConfig(BaseModel):
    model: str
    max_context_chunks: int


class ConfigSettings(BaseSettings):
    config_id:int
    ingestion: ChunkConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    generation: GenerationConfig

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(**data)


if __name__ == "__main__":  
    hp = ConfigSettings.load(Path("config", "hyperparams.json"))
    print(hp.model_dump_json())