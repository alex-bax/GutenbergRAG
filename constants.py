from enum import Enum

MAX_TOKENS = 8000
CHUNK_SIZE = 400
OVERLAP = 100
TOKEN_PR_MIN = 501_000
REQUESTS_PR_MIN = 3000

MIN_SEARCH_SCORE = 0.01

ID_FRANKENSTEIN = 84
ID_DR_JEK_MR_H = 42
ID_MOBY = 2701
ID_SHERLOCK = 1661

DEF_BOOK_NAMES_TO_IDS = {
    "The Adventures of Sherlock Holmes": ID_SHERLOCK, 
    "The Strange Case of Dr. Jekyll and Mr. Hyde": ID_DR_JEK_MR_H, 
    "The Federalist Papers": 1404, 
    "Moby Dick; Or, The Whale": ID_MOBY,
    "Meditations":2680,
    "The King in Yellow":8492
}
VER_PREFIX = "v1"


class EmbeddingDimension(int, Enum):
    SMALL = 1536
    LARGE = 3072
