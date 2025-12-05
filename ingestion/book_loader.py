import requests_async
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from models.api_response_model import GBBookMeta
from models.local_gb_book_model import GBBookMetaLocal
from vector_store_utils import upload_to_index_async
from db.operations import insert_missing_book_db
from converters import gbbookmeta_to_db_obj

from settings import get_settings, Settings
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


def get_path_by_book_id_from_cache(*, book_id:int, folder_p:Path = Path("eval_data", "gb_meta_objs_by_id")) -> list[Path]:
    folder_p.mkdir(parents=True, exist_ok=True)
    # if not folder_p.is_dir():
    #     raise FileNotFoundError(f"Folder not found: {str(folder_p)}")

    lst = [file for file in folder_p.iterdir() if file.is_file() and str(book_id) == file.stem.split('_')[-1]]
    assert len(lst) < 2
    return lst


def _write_to_files(book_content:str, gb_meta:GBBookMeta) -> Path:
    local_gb_p = Path("eval_data", "gb_meta_objs_by_id", f"{make_slug_book_key(title=gb_meta.title, gutenberg_id=gb_meta.id, author=gb_meta.authors_as_str())}.txt")
    loc_gb_meta = GBBookMetaLocal(**gb_meta.model_dump(), path_to_content=local_gb_p)
    loc_gb_json = loc_gb_meta.model_dump_json(indent=4)

    assert len(loc_gb_json) > 0
    local_gb_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(local_gb_p.with_suffix(".json"), "w", encoding='utf-8') as f:
            f.write(loc_gb_json)
        
        local_book_content_p = Path("eval_data","books") / local_gb_p.name
        with open(local_book_content_p, "w", encoding="utf-8") as f:
            f.write(book_content)
    except Exception as ex:
        print(ex)
        print([p.name for p in list(local_book_content_p.parent.glob("*"))])

    return local_gb_p


#TODO - make unit test
async def upload_missing_book_ids(*, book_ids:set[int], sett:Settings, db_sess:AsyncSession) -> tuple[list[GBBookMeta], str]:
    """Upload and book ids to vector index and insert into book meta DB if missing"""
    vector_store = await sett.get_vector_store()
    missing_book_ids = await vector_store.get_missing_ids_in_store( book_ids=book_ids)

    gb_books = []
    req_lim, token_lim = sett.get_limiters()
    print(f'--- Missing book ids: {missing_book_ids}')
    mess = ""

    for b_id in missing_book_ids:
        eval_book_paths = get_path_by_book_id_from_cache(book_id=b_id)
        
        if len(eval_book_paths) == 0:
            book_content, gb_meta = await fetch_book_content_from_id(gutenberg_id=b_id)
            assert len(book_content) > 0

            local_gb_p = _write_to_files(book_content=book_content, gb_meta=gb_meta)
            mess += " Wrote book to cache."
            print(f"GB meta obj not found in cache - fetching from Gutendex. Wrote content + gb obj to: {local_gb_p.name}")
        else:
            gb_meta = _load_gb_meta_local(path=eval_book_paths[0])
            with open(Path("eval_data", "books", eval_book_paths[0].with_suffix(".txt").name), "r", encoding="utf-8") as f:
                book_content = f.read()
            mess += f"Loaded content from cache for book id {b_id}"
            print(mess)

        print(f"*** Uploading Book id {b_id} to index")
        await upload_to_index_async(vec_store=vector_store, 
                                    embed_client=sett.get_async_emb_client(),
                                    token_limiter=token_lim,
                                    request_limiter=req_lim,
                                    raw_book_content=book_content,
                                    book_meta=gb_meta
                                )
        db_book = gbbookmeta_to_db_obj(gbm=gb_meta)
        mess += await insert_missing_book_db(db_book, db_sess)

        gb_books.append(gb_meta)

    return gb_books, mess


async def _fetch_gutendex_meta_from_id(*, gb_id:int) -> GBBookMeta:
    url = f"https://gutendex.com/books/{gb_id}/"
    resp = await requests_async.get(url)
    resp.raise_for_status()
    
    body = resp.json()#["results"]
    return GBBookMeta(**body)

    
async def gutendex_book_urls(*, n=25, languages:list[str]=["en"], text_format="text/html") -> list[dict[str, str|int|list[str]]]:
    out = []
    txt_url = "https://gutendex.com/books"
    params = {"languages": ",".join(languages)}
    
    while len(out) < n and txt_url:
        resp = await requests_async.get(txt_url, params=params if "books" in txt_url else None, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        for b in data["results"]:
            # pick a text/plain URL if present
            formats = b.get("formats", {})
            txt_url = None
            
            for k, url_v in formats.items():
                if k.startswith(text_format):
                    txt_url = url_v; break
            
            if txt_url:
                out.append({"id": b["id"], 
                            "title": b["title"], 
                            "download_url": txt_url, 
                            "authors":b["authors"]})
                
                if len(out) >= n: break

        txt_url = data.get("next")
        params = None  # only send params on first page
    return out



async def download_or_load_from_cache(*, book_key:str, url:str) -> str:
    book_p = Path("books", f'{book_key}.txt')

    if book_p.exists():
        with open(book_p, 'r', encoding="utf-8") as f:
            print(f"\n From file: {book_p.name}")
            book_txt = f.read()
        
    else:
        book_txt = await _fetch_book_content(download_url=url)

        with open(book_p, "w", encoding="utf-8") as f:
            f.write(book_txt)

    return book_txt


async def try_book_loader():
    books = await gutendex_book_urls(n=10, text_format="text/plain")
    for b in books[:5]:
        print(b["id"], b["title"], b["download_url"])

if __name__ == "__main__":
    asyncio.run(try_book_loader())
