from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from db.vector_store_abstract import VectorStore
# from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from pyrate_limiter import Limiter, Rate, Duration, InMemoryBucket, BucketAsyncWrapper
from constants import TOKEN_PR_MIN, REQUESTS_PR_MIN


# TODO: merge constants into settings
# TODO: add input extention type, e.g. whether it's html, txt, etc.
        # TODO: for each extraction type, use a different pre-processing with Strategy design pattern

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

    INDEX_NAME: str = "moby"
    EMBED_MODEL_DEPLOYMENT:str
    LLM_MODEL_DEPLOYMENT:str

    model_config = SettingsConfigDict(
        env_file=".env",  # local dev
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Postgres DB
    DB_NAME:str
    DB_PW:str
    DB_PORT:int


    def get_search_client(self) -> VectorStore:
        return VectorStore(
            endpoint=self.AZURE_SEARCH_ENDPOINT,
            index_name=self.INDEX_NAME,
            credential=AzureKeyCredential(self.AZURE_SEARCH_KEY)
        )

    def get_index_client(self) -> SearchIndexClient:
        return  SearchIndexClient(endpoint=self.AZURE_SEARCH_ENDPOINT,
                                credential=AzureKeyCredential(self.AZURE_SEARCH_KEY))

    def get_llm_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            azure_endpoint=self.AZ_OPENAI_GPT_ENDPOINT,
            api_version="2025-04-01-preview",
            api_key=self.AZ_OPENAI_GPT_KEY
        )

    def get_emb_client(self) -> AzureOpenAI:
        return AzureOpenAI(azure_endpoint=self.AZ_OPENAI_EMBED_ENDPOINT,
                                api_version="2024-12-01-preview",
                                api_key=self.AZ_OPENAI_EMBED_KEY)
    
    def make_limiters(self) -> list[Limiter]:
        REQ_RATE = Rate(REQUESTS_PR_MIN, Duration.MINUTE)              # 3,000 requests per minute
        TOK_RATE = Rate(TOKEN_PR_MIN, Duration.MINUTE)                 # 501,000 tokens per minute

        req_bucket = BucketAsyncWrapper(InMemoryBucket([REQ_RATE]))
        tok_bucket = BucketAsyncWrapper(InMemoryBucket([TOK_RATE]))
        
        req_limiter = Limiter(req_bucket)
        tok_limiter = Limiter(tok_bucket)

        return [req_limiter, tok_limiter]


@lru_cache
def get_settings() -> Settings:
    return Settings()       # type:ignore



if __name__ == "__main__":
    s = get_settings()
    print(s.AZURE_SEARCH_ENDPOINT, s.INDEX_NAME)