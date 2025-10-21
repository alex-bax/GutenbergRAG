import requests, re
from pathlib import Path


def _fetch_book(*, download_url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt") -> str:
    """
    Fetches book. Default url is gutenberg book link to Moby Dick
    """
    r = requests.get(download_url, timeout=60)
    r.raise_for_status()
    return r.text


def gutendex_book_urls(n=25, languages:list[str]=["en"], mime="text/plain") -> list[dict[str, str|int]]:
    out = []
    txt_url = "https://gutendex.com/books"
    params = {"languages": ",".join(languages)}
    while len(out) < n and txt_url:
        resp = requests.get(txt_url, params=params if "books" in txt_url else None, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for b in data["results"]:
            # pick a text/plain URL if present
            fmts = b.get("formats", {})
            txt_url = None
            # prefer UTF-8 plain text if available
            for k, url_v in fmts.items():
                if k.startswith("text/plain"):
                    txt_url = url_v; break
            if txt_url:
                out.append({"id": b["id"], "title": b["title"], "download_url": txt_url, "authors":b["authors"]})
                if len(out) >= n: break
        txt_url = data.get("next")
        params = None  # only send params on first page
    return out


def _strip_gutenberg_header_footer(*, book:str) -> str:
    start = re.search(r"^CHAPTER 1\.", book, re.M)
    end = re.search(r"End of the Project Gutenberg EBook of", book)

    book = book[start.start(): end.start()] if start and end else book
    return book.strip()


def download_or_load_from_cache(*, book_key:str, url:str) -> str:
    book_p = Path("books", f'{book_key}.txt')

    if book_p.exists():
        with open(book_p, 'r', encoding="utf-8") as f:
            print(f"Loaded: {book_p.name}")
            book_txt = f.read()
        
    else:
        book_txt = _fetch_book(download_url=url)
        book_txt = _strip_gutenberg_header_footer(book=book_txt)

        with open(book_p, "w", encoding="utf-8") as f:
            f.write(book_txt)

    return book_txt



if __name__ == "__main__":
    books = gutendex_book_urls(50)
    for b in books[:5]:
        print(b["id"], b["title"], b["download_url"])
