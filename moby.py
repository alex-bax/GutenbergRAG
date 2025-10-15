import os, re, math, requests, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields
)

from search_handler import create_missing_search_index, is_book_in_index, upload_to_index

# load_dotenv()
# TODO: do this more elegantly
AZ_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZ_SEARCH_API_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "moby")

AOAI_EP = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBED_DEPLOY = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
CHAT_DEPLOY = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# TODO: add hyper params

def main() -> None:
    index_client = SearchIndexClient(
        endpoint=AZ_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZ_SEARCH_API_KEY)
    )
    
    book_name = "Moby-Dick"
    INDEX = "moby"

    create_missing_search_index(book_index_name="moby", search_index_client=index_client)

    search_client = SearchClient(endpoint=AZ_SEARCH_ENDPOINT, 
                                 index_name=INDEX, 
                                 credential=AzureKeyCredential(AZ_SEARCH_API_KEY))
    

    if not is_book_in_index(search_client=search_client, book_name=book_name):
        upload_to_index(search_client=search_client)
    else:
        print(f"{book_name} already in index {INDEX}")

    




