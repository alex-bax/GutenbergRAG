import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AzureOpenAI
from pyrate_limiter import Limiter, Rate, Duration, InMemoryBucket, BucketAsyncWrapper

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from preprocess_book import make_slug_book_key
from search_handler import create_missing_search_index, is_book_in_index, upload_to_index_async
from retrieve import answer
from settings import get_settings
from load_book import gutendex_book_urls

from constants import TOKEN_PR_MIN, REQUESTS_PR_MIN

# TODO: add hyper params to settings
# TODO: split entire app into ingestion / retrieval

def _make_limiters() -> list[Limiter]:
    REQ_RATE = Rate(REQUESTS_PR_MIN, Duration.MINUTE)              # 3,000 requests per minute
    TOK_RATE = Rate(TOKEN_PR_MIN, Duration.MINUTE)                 # 501,000 tokens per minute

    req_bucket = BucketAsyncWrapper(InMemoryBucket([REQ_RATE]))
    tok_bucket = BucketAsyncWrapper(InMemoryBucket([TOK_RATE]))
    
    tok_limiter = Limiter(tok_bucket)
    req_limiter = Limiter(req_bucket)

    # req_limiter = Limiter(REQ_RATE)
    # tok_limiter = Limiter(TOK_RATE)

    return [req_limiter, tok_limiter]


async def main() -> None:
    ### Ingestion
    sett = get_settings()
    
    INDEX = sett.INDEX_NAME

    query = "Who's Ishmael?"

    gutenberg_book_metadata_list = gutendex_book_urls(n=10, languages=["en"], text_format="text/plain")

    books_to_download:list[dict[str,str|int]] = []

    #TODO: consider combining the two loops into one

    for b_meta in gutenberg_book_metadata_list:
        books_to_download.append({
            "title": b_meta["title"],
            "authors": "; ".join([author["name"] for author in b_meta["authors"]]), # type: ignore
            "gb_id": b_meta["id"],
            "url": b_meta["download_url"]
        })

    index_client = SearchIndexClient(endpoint=sett.AZURE_SEARCH_ENDPOINT,
                                    credential=AzureKeyCredential(sett.AZURE_SEARCH_KEY))
    
    create_missing_search_index(book_index_name="moby", 
                                search_index_client=index_client)

    search_client = SearchClient(endpoint=sett.AZURE_SEARCH_ENDPOINT, 
                                 index_name=INDEX, 
                                 credential=AzureKeyCredential(sett.AZURE_SEARCH_KEY))

    emb_client = AzureOpenAI(azure_endpoint=sett.AZ_OPENAI_EMBED_ENDPOINT,
                            api_version="2024-12-01-preview",
                            api_key=sett.AZ_OPENAI_EMBED_KEY)

    req_limiter, tok_limiter = _make_limiters()

    for b in tqdm(books_to_download):
        b["book_key"] = make_slug_book_key(title=b["title"],            # type: ignore
                                        gutenberg_id=b["gb_id"],        # type: ignore
                                        author=b["authors"],            # type: ignore
                                        lang="en")

        if not is_book_in_index(search_client=search_client, book_key=b["book_key"]):

            chapters_added = await upload_to_index_async(search_client=search_client, 
                                            embed_client=emb_client,
                                            book=b,                  # type:ignore
                                            token_limiter=tok_limiter,
                                            request_limiter=req_limiter
                                            )
        else:
            print(f"\n Already in index {INDEX} - {b['title']}")

    ### Retrieval
    # print(f'Answering the query: {query}')

    # llm_client = AzureOpenAI(azure_endpoint=sett.AZ_OPENAI_GPT_ENDPOINT,
    #                         api_version="2025-04-01-preview",
    #                         api_key=sett.AZ_OPENAI_GPT_KEY)

    # ans = answer(query=query, search_client=search_client, embed_client=emb_client, llm_client=llm_client)
    # print(ans)


if __name__ == "__main__":
    asyncio.run(main())


