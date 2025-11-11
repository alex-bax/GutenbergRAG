from enum import Enum

MAX_TOKENS = 8000
OVERLAP = 100
TOKEN_PR_MIN = 501_000
REQUESTS_PR_MIN = 3000

MIN_SEARCH_SCORE = 0.5

class EmbeddingDimension(int, Enum):
    SMALL = 1536
    LARGE = 3072
