import requests, re
from pathlib import Path

from models.api_response import GBBookMeta


def _fetch_book_content(*, download_url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt") -> str:
    """
    Fetches book. Default url is gutenberg book url for Moby Dick
    """
    r = requests.get(download_url, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_book_content_from_id(*, gutenberg_id:int) -> tuple[str, GBBookMeta]:
    gb_meta = _fetch_gutendex_meta_from_id(gb_id=gutenberg_id)
    url = gb_meta.get_txt_url()

    if not url:
        raise Exception("Gutenberg book missing txt/plain url")

    return _fetch_book_content(download_url=url), gb_meta    


def _fetch_gutendex_meta_from_id(*, gb_id:int) -> GBBookMeta:
    url = f"https://gutendex.com/books/{gb_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    
    body = resp.json()#["results"]
    return GBBookMeta(**body)

    
def gutendex_book_urls(*, n=25, languages:list[str]=["en"], text_format="text/html") -> list[dict[str, str|int|list[str]]]:
    out = []
    txt_url = "https://gutendex.com/books"
    params = {"languages": ",".join(languages)}
    
    while len(out) < n and txt_url:
        resp = requests.get(txt_url, params=params if "books" in txt_url else None, timeout=60)
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




def download_or_load_from_cache(*, book_key:str, url:str) -> str:
    book_p = Path("books", f'{book_key}.txt')

    if book_p.exists():
        with open(book_p, 'r', encoding="utf-8") as f:
            print(f"\n From file: {book_p.name}")
            book_txt = f.read()
        
    else:
        book_txt = _fetch_book_content(download_url=url)

        with open(book_p, "w", encoding="utf-8") as f:
            f.write(book_txt)

    return book_txt



if __name__ == "__main__":
    books = gutendex_book_urls(n=10, text_format="text/plain")
    for b in books[:5]:
        print(b["id"], b["title"], b["download_url"])
