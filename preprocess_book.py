import backoff
from pathlib import Path
import re, unicodedata, tiktoken
from openai import AzureOpenAI
from openai._exceptions import RateLimitError
from tiktoken import Encoding

from data_classes.vector_db import EmbeddingVec
from constants import EmbeddingDimension, MAX_TOKENS, OVERLAP

def extract_txt(*, raw_book: str) -> str:
    start_match = re.search(pattern=r'\*\*\*\s?START OF TH(IS|E) PROJECT GUTENBERG EBOOK.', string=raw_book)
    end_match = re.search(pattern=r'\*\*\* end of the project gutenberg ebook', string=raw_book.lower())

    return raw_book[start_match.end():end_match.start()] if start_match and end_match else ""


@backoff.on_exception(backoff.expo, RateLimitError, max_time=120, max_tries=6)
def create_embeddings(*, embed_client:AzureOpenAI, model_deployed:str, texts:list[str]) -> list[EmbeddingVec]:
    resp = embed_client.embeddings.create(
        input=texts,
        model=model_deployed,
    )
    return [EmbeddingVec(vector=emb_obj.embedding, dim=EmbeddingDimension.SMALL) for emb_obj in resp.data]


def _count_tokens(text: str, enc:Encoding) -> int:
    return len(enc.encode(text))

def _batch_texts_by_tokens(texts: list[str],
                           max_tokens_per_request: int=MAX_TOKENS) -> list[list[str]]:
    """
    Greedily packs texts into batches so that the sum of tokens per batch
    stays under max_tokens_per_request.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    batches, current, sum_current_tokens = [], [], 0
    for t in texts:
        token_len = _count_tokens(t, enc)
        if current and sum_current_tokens + token_len > max_tokens_per_request:
            batches.append(current)
            current, sum_current_tokens = [t], token_len
        else:
            current.append(t)
            sum_current_tokens += token_len
    
    if current:
        batches.append(current)

    return batches




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
