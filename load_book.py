import requests
from pathlib import Path


def fetch_book(*, book_download_url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt") -> str:
    """
    Fetches book. Default url is gutenberg book link to Moby Dick
    """
    r = requests.get(book_download_url, timeout=60)
    r.raise_for_status()
    return r.text