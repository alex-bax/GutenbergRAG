from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from db.database import engine, get_db_session_factory
from db.database import Base
from config.settings import Settings
from ingestion.book_loader import upload_missing_book_ids
from stats import make_collection_fingerprint
import matplotlib
matplotlib.use("Agg")
from datetime import datetime

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
        seed_ids = list(hp_ing.default_ids_used.values())#[:1]       

    now = datetime.now().strftime("%d-%m-%Y_%H%M")

    # Seed the vector store
    print(f'DEF GB SEEDS: {seed_ids}')
    await settings.get_vector_store()
    db_factory = get_db_session_factory()
    _, _, book_stats = await upload_missing_book_ids(book_ids=set(seed_ids), 
                                                    sett=settings, 
                                                    time_started=now,
                                                    db_factory=db_factory)

    hp = settings.get_hyperparams()

    if len(book_stats) > 0:
        try:
            collection_finger = make_collection_fingerprint(chunk_stats=book_stats, 
                                                            config_id_used=hp.config_id)

            with open(Path("stats", "index_stats", f"conf_id_{hp.config_id}_{hp.collection}_{now}.json"), "w") as f:
                f.write(collection_finger.model_dump_json(indent=4))
        except Exception as ex:
            print(ex)
            print(book_stats)

    yield

    await engine.dispose()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="MobyRAG", 
                  swagger_ui_parameters={
                        "logo": {
                            "url": "imgs/GBRAGLogo.png",
                            "altText": "Gutenberg RAG API Logo"
                        }
                    },
                  lifespan=lifespan)
    app.state.settings = settings

    # include_router(...)
    return app
