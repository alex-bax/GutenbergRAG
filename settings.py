from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict,
from pydantic import PrivateAttr
from typing import Literal
from db.vector_store_abstract import AsyncVectorStore

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI, AzureOpenAI
from pyrate_limiter import Limiter, Rate, Duration, InMemoryBucket, BucketAsyncWrapper
from constants import TOKEN_PR_MIN, REQUESTS_PR_MIN, EmbeddingDimension


# TODO: merge constants into settings
# TODO: separate this into multiple Settings, e.g. for DB, vector store, etc.

# Initializes fields via .env file

class Settings(BaseSettings):
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_EMBED_KEY: str

    AZ_OPENAI_EMBED_ENDPOINT: str
    AZ_OPENAI_EMBED_KEY: str

    AZ_OPENAI_GPT_ENDPOINT: str
    AZ_OPENAI_GPT_KEY: str

    QDRANT_SEARCH_ENDPOINT: str
    QDRANT_SEARCH_KEY: str

    VECTOR_STORE_TO_USE: Literal["Qdrant", "AzureAiSearch"] = "Qdrant"
    INDEX_NAME: str = "gutenberg"
    EMBED_MODEL_DEPLOYMENT:str
    LLM_MODEL_DEPLOYMENT:str

    # Postgres DB
    DB_NAME:str
    DB_PW:str
    DB_PORT:int

    EMBEDDING_DIM:EmbeddingDimension = EmbeddingDimension.SMALL
   
   # Not to be validated as model fields
    _llm_client: AzureOpenAI | None = PrivateAttr(default=None)
    _emb_client: AsyncAzureOpenAI | None = PrivateAttr(default=None)
    _vector_store: AsyncVectorStore | None = PrivateAttr(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",  # local dev
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    async def get_vector_store(self) -> AsyncVectorStore:
        from db.az_search_vector_store import AzSearchVectorStore
        from db.qdrant_vector_store import QdrantVectorStore
        
        if self._vector_store is None:
            if self.VECTOR_STORE_TO_USE == "Qdrant":
                qdrant_v_store = QdrantVectorStore(settings=self, collection_name=self.INDEX_NAME)
                await qdrant_v_store.initialize()
                self._vector_store = qdrant_v_store
            elif self.VECTOR_STORE_TO_USE == "AzureAiSearch":
                self._vector_store = AzSearchVectorStore(settings=self)
            else:
                raise ValueError("No valid Vector store specified - Check settings!")
        
        return self._vector_store


    def get_llm_client(self) -> AzureOpenAI:
        if self._llm_client is None:
            self._llm_client = AzureOpenAI(azure_endpoint=self.AZ_OPENAI_GPT_ENDPOINT,
                                            api_version="2025-04-01-preview",
                                            api_key=self.AZ_OPENAI_GPT_KEY)
        return self._llm_client

    def get_emb_client(self) -> AsyncAzureOpenAI:
        if self._emb_client is None:
            self._emb_client = AsyncAzureOpenAI(azure_endpoint=self.AZ_OPENAI_EMBED_ENDPOINT,
                                                api_version="2024-12-01-preview",
                                                api_key=self.AZ_OPENAI_EMBED_KEY)
        return self._emb_client
    
    def make_limiters(self) -> list[Limiter]:
        REQ_RATE = Rate(REQUESTS_PR_MIN, Duration.MINUTE)              # 3,000 requests per minute
        TOK_RATE = Rate(TOKEN_PR_MIN, Duration.MINUTE)                 # 501,000 tokens per minute

        req_bucket = BucketAsyncWrapper(InMemoryBucket([REQ_RATE]))
        tok_bucket = BucketAsyncWrapper(InMemoryBucket([TOK_RATE]))
        
        req_limiter = Limiter(req_bucket)
        tok_limiter = Limiter(tok_bucket)

        return [req_limiter, tok_limiter]


@lru_cache      # Enforces singleton pattern - only one settings instance allowed
def get_settings() -> Settings:
    return Settings()       # type:ignore



if __name__ == "__main__":
    s = get_settings()
    print(s.AZURE_SEARCH_ENDPOINT, s.INDEX_NAME)