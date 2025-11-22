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
from tqdm import tqdm

from openai import AsyncAzureOpenAI


# TODO: make this async also
@backoff.on_exception(wait_gen=backoff.expo, exception=RateLimitError, max_time=120, max_tries=6)
async def _create_embeddings(*, embed_client:AsyncAzureOpenAI, 
                       model_deployed:str, 
                       batches:list[str]) -> list[EmbeddingVec]:
    resp = await embed_client.embeddings.create(
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


async def create_embeddings_async(*, embed_client:AsyncAzureOpenAI, 
                                    model_deployed: str, 
                                    inp_batches: list[list[str]], 
                                    tok_limiter:Limiter, 
                                    req_limiter:Limiter) -> list[EmbeddingVec]:
    """Create async Azure embeddings with built-in rate limiting and graceful backoff."""
    all_embeddings = []
    enc_ = tiktoken.get_encoding("cl100k_base")

    for batch in inp_batches:
        tokens_needed = sum([_count_tokens(chunk, enc=enc_) for chunk in batch])
        print(f'\nEmbedding: tokens needed from limiter: {tokens_needed}')
        
        await _acquire_budget_async(tok_limiter=tok_limiter, 
                            req_limiter=req_limiter, 
                            tokens_needed=tokens_needed) 

        # TODO - change to use async
        embs = await _create_embeddings(embed_client=embed_client, 
                                  model_deployed=model_deployed, 
                                  batches=batch)
        
        all_embeddings.extend(embs)
        
    return all_embeddings
