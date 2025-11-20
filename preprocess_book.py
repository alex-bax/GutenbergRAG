import backoff, time, asyncio
from pathlib import Path
import re, unicodedata, tiktoken
from openai import AzureOpenAI
from openai._exceptions import RateLimitError
from tiktoken import Encoding

from models.vector_db_model import EmbeddingVec
from constants import EmbeddingDimension, MAX_TOKENS, OVERLAP
from openai import RateLimitError
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException
import pandas as pd
from tqdm import tqdm


def clean_headers(*, raw_book: str) -> str:
    start_match = re.search(pattern=r'\*\*\*\s?START OF TH(IS|E) PROJECT GUTENBERG EBOOK', string=raw_book)
    end_match = re.search(pattern=r'\*\*\* end of the project gutenberg ebook', string=raw_book.lower())

    return raw_book[start_match.end():end_match.start()] if start_match and end_match else ""


@backoff.on_exception(wait_gen=backoff.expo, exception=RateLimitError, max_time=120, max_tries=6)
def create_embeddings(*, embed_client:AzureOpenAI, 
                       model_deployed:str, 
                       batches:list[str]) -> list[EmbeddingVec]:
    resp = embed_client.embeddings.create(
        input=batches,
        model=model_deployed,
    )
    return [EmbeddingVec(vector=emb_obj.embedding, dim=EmbeddingDimension.SMALL) for emb_obj in resp.data]



def _count_tokens(text: str, enc:Encoding) -> int:
    return len(enc.encode(text))

def batch_texts_by_tokens(*, texts: list[str],
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

async def _acquire_budget_async(*, tok_limiter:Limiter, req_limiter:Limiter, tokens_needed: int, identity: str = "embeddings"):
    """Non-blocking: awaits until both token & request budgets allow the call."""
    while True:
        try:
            await tok_limiter.try_acquire_async(f"{identity}_tpm", weight=tokens_needed)
            await req_limiter.try_acquire_async(f"{identity}_rpm", weight=1)
            return  # allowed
        except BucketFullException as ex:
            sleep_interval_secs = float(ex.rate.interval) / 1000        # precise wait suggested by the limiter
            print(f"** Exceeding budget - async sleeping {sleep_interval_secs} secs")
            await asyncio.sleep(sleep_interval_secs)


async def create_embeddings_async(*, embed_client:AzureOpenAI, 
                              model_deployed: str, 
                              inp_batches: list[list[str]], 
                              tok_limiter:Limiter, 
                              req_limiter:Limiter) -> list[EmbeddingVec]:
    """Call Azure embeddings with built-in rate limiting and graceful backoff."""
    all_embeddings = []
    enc_ = tiktoken.get_encoding("cl100k_base")

    for batch in inp_batches:
        tokens_needed = sum([_count_tokens(chunk, enc=enc_) for chunk in batch])
        print(f'\ntokens needed from limiter: {tokens_needed}')
        
        await _acquire_budget_async(tok_limiter=tok_limiter, 
                            req_limiter=req_limiter, 
                            tokens_needed=tokens_needed) 

        # TODO - change to use async
        embs = create_embeddings(embed_client=embed_client, 
                                  model_deployed=model_deployed, 
                                  batches=batch)
        
        all_embeddings.extend(embs)
        
    return all_embeddings


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
