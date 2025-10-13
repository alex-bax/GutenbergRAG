import requests, re
from pathlib import Path


def _fetch_book(*, book_download_url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt") -> str:
    """
    Fetches book. Default url is gutenberg book link to Moby Dick
    """
    r = requests.get(book_download_url, timeout=60)
    r.raise_for_status()
    return r.text


def _strip_gutenberg_header_footer(*, book:str) -> str:
    start = re.search(r"^CHAPTER 1\.", book, re.M)
    end = re.search(r"End of the Project Gutenberg EBook of", book)

    book = book[start.start(): end.start()] if start and end else book
    return book.strip()


def download_or_load_from_cache(*, book_path:Path) -> str:
    if book_path.exists():
        with open(book_path, 'r', encoding="utf-8") as f:
            print(f"Loaded: {book_path.name}")
            moby_book = f.read()
        
    else:
        moby_book = _fetch_book()
        moby_book = _strip_gutenberg_header_footer(book=moby_book)

        with open(book_path, "w", encoding="utf-8") as f:
            f.write(moby_book)

    return moby_book