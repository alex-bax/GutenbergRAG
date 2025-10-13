import os, re, math, requests, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

import tiktoken
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

# load_dotenv()

SEARCH_EP = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "moby")

AOAI_EP = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBED_DEPLOY = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
CHAT_DEPLOY = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]



