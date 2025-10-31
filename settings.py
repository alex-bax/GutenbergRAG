from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    INDEX_NAME: str = "moby"

    EMBED_MODEL_DEPOYED: str

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


@lru_cache
def get_settings() -> Settings:
    return Settings()       # type:ignore



if __name__ == "__main__":
    s = get_settings()
    print(s.AZURE_SEARCH_ENDPOINT, s.INDEX_NAME)