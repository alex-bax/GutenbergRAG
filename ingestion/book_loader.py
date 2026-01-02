from typing import Callable
import requests_async
import json
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from tqdm import tqdm
from db.database import DbSessionFactory, get_async_db_sess, open_session
from db.operations import insert_missing_book_db
from models.api_response_model import GBBookMeta
from models.local_gb_book_model import GBBookMetaLocal
from vector_store_utils import async_upload_book_to_index

from stats import make_collection_fingerprint
from converters import gbbookmeta_to_db_obj
from models.schema import DBBookChunkStats, DBBookMetaData
from config.settings import Settings
from ingestion.preprocess_book import make_slug_book_key

# TODO: make async
async def _fetch_book_content(*, download_url) -> str:
    resp = await requests_async.get(download_url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return resp.text

async def fetch_book_content_from_id(*, gutenberg_id:int) -> tuple[str, GBBookMeta]:
    gb_meta = await _fetch_gutendex_meta_from_id(gb_id=gutenberg_id)
    url = gb_meta.get_new_txt_url(gutenberg_id) #gb_meta.get_txt_url()

    if not url:
        raise Exception("Gutenberg book missing its txt/plain url")

    content = await _fetch_book_content(download_url=url)
    return content, gb_meta    

def _load_gb_meta_local(*, path: Path) -> GBBookMetaLocal:
    json_text = Path(path).read_text(encoding="utf-8")
    return GBBookMetaLocal.model_validate_json(json_text)


def get_cached_paths_by_book_id(*, book_id:int, folder_p:Path) -> list[Path]:
    lst = [file for file in folder_p.iterdir() if file.is_file() and str(book_id) == file.stem.split('_')[-1] and file.suffix == '.json']
    return lst


def _write_to_files(book_content:str, gb_meta:GBBookMeta) -> Path:
    loc_gb_p = Path("evals", "books", f"{make_slug_book_key(title=gb_meta.title, gutenberg_id=gb_meta.id, author=gb_meta.authors_as_str())}.txt")
    loc_gb_p.parent.mkdir(parents=True, exist_ok=True)
    
    loc_gb_meta = GBBookMetaLocal(**gb_meta.model_dump(), path_to_content=loc_gb_p)
    gb_json = loc_gb_meta.model_dump_json(indent=4)

    assert len(gb_json) > 0

    try:
        with open(loc_gb_p, "w", encoding="utf-8") as f:
            f.write(book_content)
        
        with open(loc_gb_p.with_suffix(".json"), "w", encoding='utf-8') as f:
            f.write(gb_json)
    except Exception as ex:
        print(f"!! DEBUG TRACE local_book_content_p:{str(loc_gb_p)}")
        print(f"local_gb_p {loc_gb_p}")

    return loc_gb_p


#TODO - make unit test
async def upload_missing_book_ids(*, book_ids:set[int], 
                                  sett:Settings, 
                                  db_factory: DbSessionFactory,
                                  time_started:str
                                ) -> tuple[list[GBBookMeta], str, list[DBBookChunkStats]]:
    """Upload and book ids to vector index and insert into book meta DB if missing"""
    vector_store = await sett.get_vector_store()
    missing_book_ids = await vector_store.get_missing_ids_in_store( book_ids=book_ids)

    gb_books = []
    req_lim, token_lim = sett.get_limiters()
    print(f'--- Missing book ids: {missing_book_ids}')
    mess = ""
    cache_p = Path("evals", "books")
    cache_p.mkdir(parents=True, exist_ok=True)

    book_stats = []

    for b_id in tqdm(missing_book_ids): 
        eval_book_paths = get_cached_paths_by_book_id(book_id=b_id, folder_p=cache_p)
        
        if len(eval_book_paths) == 0:
            book_content, gb_meta = await fetch_book_content_from_id(gutenberg_id=b_id)
            assert len(book_content) > 0

            local_gb_p = _write_to_files(book_content=book_content, gb_meta=gb_meta)
            mess += " Wrote book to cache."
            print(f"GB meta obj not found in cache - fetching from Gutendex. Wrote content + gb obj to: {local_gb_p.name}")
        else:
            gb_meta = _load_gb_meta_local(path=eval_book_paths[0])
            gb_meta.path_to_content.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(gb_meta.path_to_content, "r", encoding="utf-8") as f:
                    book_content = f.read()
                
                # book_content = book_content[:2000] if sett.is_test else book_content
                # book_content = book_content[:1000] if True else book_content
                mess += f"\nLoaded content from cache for book id {b_id}"
                print(mess) 
            except Exception as exc:
                print(f"EXC: tried {str(gb_meta.path_to_content)} {exc}")
                
        print(f"*** Uploading Book id {b_id} to index")

        upload_chunks, db_b_stats = await async_upload_book_to_index(vec_store=vector_store, 
                                                                    embed_client=sett.get_async_emb_client(),
                                                                    token_limiter=token_lim,
                                                                    request_limiter=req_lim,
                                                                    raw_book_content=book_content,
                                                                    book_meta=gb_meta,
                                                                    sett=sett,
                                                                    time_started=time_started
                                                                )
        book_stats.append(db_b_stats)
        
        db_book = gbbookmeta_to_db_obj(gbm=gb_meta)
        db_book.chunk_stats = db_b_stats
        async with open_session(db_factory) as db_sess:
            is_inserted, mess_ = await insert_missing_book_db(book_meta=db_book, 
                                                                db_sess=db_sess)
        mess += mess_

        gb_books.append(gb_meta)

 
    return gb_books, mess, book_stats


async def _fetch_gutendex_meta_from_id(*, gb_id:int) -> GBBookMeta:
    url = f"https://gutendex.com/books/{gb_id}/"
    resp = await requests_async.get(url)
    resp.raise_for_status()
    
    body = resp.json()#["results"]
    return GBBookMeta(**body)


async def download_or_load_from_cache(*, book_key:str, url:str) -> str:
    book_p = Path("evals", "books", f'{book_key}.txt')

    if book_p.exists():
        with open(book_p, 'r', encoding="utf-8") as f:
            print(f"\n From file: {book_p.name}")
            book_txt = f.read()
        
    else:
        book_txt = await _fetch_book_content(download_url=url)

        with open(book_p, "w", encoding="utf-8") as f:
            f.write(book_txt)

    return book_txt


# async def try_book_loader():
#     books = await gutendex_book_urls(n=10, text_format="text/plain")
#     for b in books[:5]:
#         print(b["id"], b["title"], b["download_url"])

# if __name__ == "__main__":
#     asyncio.run(try_book_loader())
