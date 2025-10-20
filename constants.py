from enum import Enum

MAX_TOKENS = 600
OVERLAP = 60

class EmbeddingDimension(int, Enum):
    SMALL = 1536
    LARGE = 3072
