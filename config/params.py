import json
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from pathlib import Path
from typing import Literal
from functools import lru_cache

VER_PREFIX = "v1"

class EmbeddingDimension(int, Enum):
    SMALL = 1536
    LARGE = 3072

class IngestionConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 100
    chunk_strategy:Literal["fixed", "semantic"]
    embed_model:str
    embed_dim:EmbeddingDimension = EmbeddingDimension.SMALL
    default_ids_used:dict[str,int]
    requests_pr_min:int
    tokens_pr_min:int
    max_tokens_pr_req:int
    sem_split_break_percentile:int
    sem_split_buffer_size: int

class RetrievalConfig(BaseModel):
    top_k: int = 8
    vector_db:Literal["Qdrant", "Azure AI Search"]
    vector_db_type:Literal["vector", "hybrid_search"]
    top_k:int

class RerankConfig(BaseModel):
    enabled: bool = True
    batch_size:int
    model:str
    rank_method:str


class GenerationConfig(BaseModel):
    model: str
    num_context_chunks: int


class ConfigParamSettings(BaseSettings):
    config_id:int
    collection:str
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    generation: GenerationConfig

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(**data)
    
    
@lru_cache
def get_config(path:Path) -> ConfigParamSettings: 
    return ConfigParamSettings.load(path)


if __name__ == "__main__":  
    hp = ConfigParamSettings.load(Path("config", "hyperparams.json"))
    print(hp.model_dump_json())