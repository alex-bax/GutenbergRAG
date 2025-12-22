from fastapi import FastAPI
from contextlib import asynccontextmanager

from db.database import engine
from db.database import Base
from config.settings import Settings
from ingestion.book_loader import upload_missing_book_ids

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:      # startup, create tables
        await conn.run_sync(Base.metadata.create_all)

    settings: Settings = app.state.settings

    # Decide what to seed
    hp_ing = settings.get_hyperparams().ingestion
    if settings.is_test:
        seed_ids = {hp_ing.default_ids_used["Frankenstein; Or, The Modern Prometheus"]}
    else:
        seed_ids = set(hp_ing.default_ids_used.values())

    # Seed the vector store
    print(f'DEF GB SEEDS: {seed_ids}')
    await settings.get_vector_store()
    await upload_missing_book_ids(
                        book_ids=seed_ids,
                        sett=settings,
                    )
    yield

    await engine.dispose()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="MobyRAG", lifespan=lifespan)
    app.state.settings = settings

    # include_router(...)
    return app
