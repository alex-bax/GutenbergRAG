from pathlib import Path
import unicodedata, re

def clean_headers(*, raw_book: str) -> str:
    start_match = re.search(pattern=r'\*\*\*\s?START OF TH(IS|E) PROJECT GUTENBERG EBOOK', string=raw_book)
    end_match = re.search(pattern=r'\*\*\* end of the project gutenberg ebook', string=raw_book.lower())

    return raw_book[start_match.end():end_match.start()] if start_match and end_match else ""



def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"[-\s]+", "-", s)
        return s


# TODO: make this accept a GBBookMeta
# delete some of its attrs to make it fit: https://stackoverflow.com/questions/1120927/which-is-better-in-python-del-or-delattr
def make_slug_book_key(title: str, gutenberg_id:int, author: str, 
                       year: int|None=None, lang: str|None=None):
    parts = [_norm(title)]
    if author: parts.append(_norm(author))
    if gutenberg_id: parts.append(f"{gutenberg_id}")
    if year:   parts.append(str(year))
    if lang:   parts.append(_norm(lang))          # e.g., "en"
    return "_".join(parts)



if __name__ == "__main__":
    book_p = r"C:\Users\alext\Documents\Code\RAG\mobyRag\books\franken-stein.txt"
    with open(Path(book_p), 'r') as f:
        txt = f.read()

    d = clean_headers(raw_book=txt)

    # print(make_slug_book_key("Moby Dick", 42, "Herman Melville", 1851, "gutenberg", "en"))
    # print(make_slug_book_key("The Odyssey", gutenberg_id=42,))
