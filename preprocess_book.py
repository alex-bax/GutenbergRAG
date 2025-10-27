import re, unicodedata
from openai import AzureOpenAI
from pathlib import Path
from data_classes.vector_db import EmbeddingVec
from constants import EmbeddingDimension

def extract_txt(*, raw_book: str) -> str:
    start_match = re.search(pattern=r'\*\*\*\s?START OF TH(IS|E) PROJECT GUTENBERG EBOOK.', string=raw_book)
    end_match = re.search(pattern=r'\*\*\* end of the project gutenberg ebook', string=raw_book.lower())

    return raw_book[start_match.end():end_match.start()] if start_match and end_match else ""

# def extract_html(*, html_book:str) -> dict[int, HtmlChapter]:
#     bs = BeautifulSoup(markup=html_book, features="lxml")
#     chapters:ResultSet[Tag] = bs.find_all(attrs={"class": "chapter"})

#     if len(chapters) == 0:
#         # TODO: book doesn't use chapter structure - instead use href-approach
        

#         return {}

#     extr_html_chs = {}
#     for i, ch in enumerate(chapters, start=0):
#         txt = ch.get_text(separator=" ", strip=True)
#         extr_html_chs[i] = HtmlChapter(content=txt, tag=ch.name, attrs=ch.attrs)        # type:ignore ; 0th chapter is often the contents table or similar

#     return extr_html_chs


def create_embeddings(*, embed_client:AzureOpenAI, model_deployed:str, texts:list[str]) -> list[EmbeddingVec]:
    resp = embed_client.embeddings.create(
        input=texts,
        model=model_deployed
    )

    return [EmbeddingVec(vector=emb_obj.embedding, dim=EmbeddingDimension.SMALL) for emb_obj in resp.data]
    


def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"[-\s]+", "-", s)
        return s

def make_slug_book_key(title: str, gutenberg_id:int, author: str|None=None, year: int|None=None, source: str|None=None, lang: str|None=None):
    parts = [_norm(title)]
    if author: parts.append(_norm(author))
    if gutenberg_id: parts.append(f"{gutenberg_id}")
    if year:   parts.append(str(year))
    if source: parts.append(_norm(source))        # e.g., "gutenberg"
    if lang:   parts.append(_norm(lang))          # e.g., "en"
    return "_".join(parts)



if __name__ == "__main__":
    book_p = r"C:\Users\alext\Documents\Code\RAG\mobyRag\books\franken-stein.txt"
    with open(Path(book_p), 'r') as f:
        txt = f.read()

    d = extract_txt(raw_book=txt)

    # print(make_slug_book_key("Moby Dick", 42, "Herman Melville", 1851, "gutenberg", "en"))
    # print(make_slug_book_key("The Odyssey", gutenberg_id=42,))
