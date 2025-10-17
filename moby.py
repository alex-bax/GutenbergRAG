import os, re, math, requests, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields
)

from search_handler import create_missing_search_index, is_book_in_index, upload_to_index
from retrieve import answer

# load_dotenv()
# TODO: do this more elegantly
AZ_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZ_SEARCH_API_KEY = os.environ["AZURE_SEARCH_KEY"]

AZ_OPENAI_EMBED_ENDPOINT = os.environ["AZ_OPENAI_EMBED_ENDPOINT"]
AZ_OPENAI_EMBED_KEY = os.environ["AZ_OPENAI_EMBED_KEY"]

AZ_OPENAI_GPT_ENDPOINT = os.environ["AZ_OPENAI_GPT_ENDPOINT"]
AZ_OPENAI_GPT_KEY = os.environ["AZ_OPENAI_GPT_KEY"]


# TODO: add hyper params

def main() -> None:
    index_client = SearchIndexClient(endpoint=AZ_SEARCH_ENDPOINT,
                                    credential=AzureKeyCredential(AZ_SEARCH_API_KEY))
    
    book_name = "Moby-Dick"
    INDEX = "moby"

    query = "Who's Ishmael?"

    create_missing_search_index(book_index_name="moby", 
                                search_index_client=index_client)

    search_client = SearchClient(endpoint=AZ_SEARCH_ENDPOINT, 
                                 index_name=INDEX, 
                                 credential=AzureKeyCredential(AZ_SEARCH_API_KEY))

    emb_client = AzureOpenAI(azure_endpoint=AZ_OPENAI_EMBED_ENDPOINT,
                            api_version="2024-12-01-preview",
                            api_key=AZ_OPENAI_EMBED_KEY)

    if not is_book_in_index(search_client=search_client, book_name=book_name):
        uuids_added = upload_to_index(search_client=search_client, 
                                      embed_client=emb_client)
        print(uuids_added)
    else:
        print(f"{book_name} already in index {INDEX}")

    print(f'Answering query: {query}')

    llm_client = AzureOpenAI(azure_endpoint=AZ_OPENAI_GPT_ENDPOINT,
                            api_version="2024-12-01-preview",
                            api_key=AZ_OPENAI_GPT_KEY)

    answer(query=query, search_client=search_client, embed_client=emb_client, llm_client=llm_client)


if __name__ == "__main__":
    main()


