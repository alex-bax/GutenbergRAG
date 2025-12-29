from pathlib import Path
from functools import lru_cache
from config.params import get_config, ConfigParamSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PrivateAttr, Field
from typing import Literal
from db.fake_vector_store import InMemoryVectorStore
from db.vector_store_abstract import AsyncVectorStore

from openai import AsyncAzureOpenAI, AzureOpenAI
from pyrate_limiter import Limiter, Rate, Duration, InMemoryBucket, BucketAsyncWrapper
# from config.hyperparams import TOKEN_PR_MIN, requests_pr_min, DEF_BOOK_NAMES_TO_IDS, ID_FRANKENSTEIN, ConfigSettings, EmbeddingDimension

# Initializes fields via .env file

class Settings(BaseSettings):
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str

    AZ_OPENAI_EMBED_ENDPOINT: str
    AZ_OPENAI_EMBED_KEY: str

    AZ_OPENAI_GPT_ENDPOINT: str
    AZ_OPENAI_GPT_KEY: str

    QDRANT_SEARCH_ENDPOINT: str
    QDRANT_SEARCH_KEY: str

    VECTOR_STORE_TO_USE: Literal["Qdrant", "AzureAiSearch"] = "Qdrant"
    
    EMBED_MODEL_DEPLOYMENT:str
    AZ_OPENAI_MODEL_DEPLOYMENT:str
    
    AZ_OPENAI_API_VER: str

    # Supabase Postgres DB
    DB_NAME:str
    DB_PW:str
    DB_USER:str
    DB_PORT:int

    is_test:bool = False
    hyperparam_path:Path
    RUN_QDRANT_TESTS:bool

   # Not to be validated as model fields
    _llm_client: AzureOpenAI | None = PrivateAttr(default=None)
    _async_emb_client: AsyncAzureOpenAI | None = PrivateAttr(default=None)
    _emb_client: AzureOpenAI | None = PrivateAttr(default=None)
    _vector_store: AsyncVectorStore | None = PrivateAttr(default=None)

    _req_limiter: Limiter | None = PrivateAttr(default=None)
    _tok_limiter: Limiter | None = PrivateAttr(default=None)
    _hyperparams:ConfigParamSettings | None = PrivateAttr(default=None)

    @property
    def active_collection(self) -> str:
        hp = self.get_hyperparams()
        return "test_" + hp.collection if self.is_test else hp.collection


    model_config = SettingsConfigDict(
        env_file=".env",  # local dev
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    async def get_vector_store(self) -> AsyncVectorStore:
        # from db.database import get_db
        # from ingestion.book_loader import upload_missing_book_ids
        from db.az_search_vector_store import AzSearchVectorStore
        from db.qdrant_vector_store import QdrantVectorStore

        if self._vector_store is None:
            if self.is_test:
                self._vector_store = InMemoryVectorStore()
            elif self.VECTOR_STORE_TO_USE == "Qdrant":
                qdrant_v_store = QdrantVectorStore(settings=self, collection_name=self.active_collection)
                self._vector_store = qdrant_v_store
            elif self.VECTOR_STORE_TO_USE == "AzureAiSearch":
                az_vec_store = AzSearchVectorStore(settings=self)
                self._vector_store = az_vec_store
            else:
                raise ValueError("No valid Vector store specified - Check settings!")
        
            await self._vector_store.create_missing_collection(collection_name=self.active_collection)

        return self._vector_store

    async def close_vector_store(self) -> None:
        """Clean up async resources – mainly for tests."""
        if self._vector_store is not None:
            from db.qdrant_vector_store import QdrantVectorStore

            if isinstance(self._vector_store, QdrantVectorStore):
                await self._vector_store.close_conn()
            self._vector_store = None


    def get_hyperparams(self) -> ConfigParamSettings:
        if self._hyperparams is None:
            self._hyperparams = get_config(path=self.hyperparam_path)
        return self._hyperparams


    def get_llm_client(self) -> AzureOpenAI:
        if self._llm_client is None:
            self._llm_client = AzureOpenAI(azure_endpoint=self.AZ_OPENAI_GPT_ENDPOINT,
                                            api_version=self.AZ_OPENAI_API_VER,
                                            api_key=self.AZ_OPENAI_GPT_KEY)
        return self._llm_client


    def get_async_emb_client(self) -> AsyncAzureOpenAI:
        if self._async_emb_client is None:
            self._async_emb_client = AsyncAzureOpenAI(azure_endpoint=self.AZ_OPENAI_EMBED_ENDPOINT,
                                                api_version="2024-12-01-preview",
                                                api_key=self.AZ_OPENAI_EMBED_KEY)
        return self._async_emb_client
    
    # Currently used for eval 
    def get_emb_client(self) -> AzureOpenAI:
        if self._emb_client is None:
            self._emb_client = AzureOpenAI(azure_endpoint=self.AZ_OPENAI_EMBED_ENDPOINT,
                                                api_version="2024-12-01-preview",
                                                api_key=self.AZ_OPENAI_EMBED_KEY)
        return self._emb_client

    
    def get_limiters(self) -> list[Limiter]:
        """Creates the limiters used for embedding if None. 
        :returns: [req_limiter, tok_limiter] 
         """
        assert self._hyperparams
        if self._req_limiter is None:
            REQ_RATE = Rate(self._hyperparams.ingestion.requests_pr_min, Duration.MINUTE)              # 3,000 requests per minute
            req_bucket = BucketAsyncWrapper(InMemoryBucket([REQ_RATE]))
            self._req_limiter = Limiter(req_bucket)

        if self._tok_limiter is None:
            TOK_RATE = Rate(self._hyperparams.ingestion.tokens_pr_min, Duration.MINUTE)                 # 501,000 tokens per minute
            tok_bucket = BucketAsyncWrapper(InMemoryBucket([TOK_RATE]))
            self._tok_limiter = Limiter(tok_bucket)

        return [self._req_limiter, self._tok_limiter]


@lru_cache      # Enforces singleton pattern - only one settings instance allowed
def get_settings(is_test=False, 
                 hyperparam_p=Path("config","hp-sem70p-ch.json")) -> Settings:
    sett = Settings(is_test=is_test, 
                    hyperparam_path=hyperparam_p)       # type:ignore
    sett.get_hyperparams()
    return sett



if __name__ == "__main__":
    s = get_settings()
    print(s.AZURE_SEARCH_ENDPOINT, s.active_collection)