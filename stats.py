import numpy as np
from pydantic import BaseModel
from models.schema import DBBookChunkStats


class CollectionFingerprint(BaseModel):
    config_id_used: int | None
    book_count: int
    total_chunks: int

    # Book-level summary (median, p90)
    book_chunk_count_median: float
    book_chunk_count_p90: float

    book_token_mean_median: float
    book_token_mean_p90: float

    book_token_std_median: float
    book_token_std_p90: float

    book_token_max_median: float
    book_token_max_p90: float

    # Chunk-level token_count pooled (p10, p50, p90, p99)
    chunk_token_p10: float
    chunk_token_p50: float
    chunk_token_p90: float
    chunk_token_p99: float

    # Outlier rates
    pct_books_token_std_gt_p90: float
    pct_books_token_max_gt_2xp90: float
    pct_books_chunk_count_gt_p99: float


def make_collection_fingerprint(
    *,
    chunk_stats: list[DBBookChunkStats],
    config_id_used: int|None = None,
) -> CollectionFingerprint:
   
    if not chunk_stats:
        raise ValueError("No book_chunk_stats rows provided")

    if config_id_used is not None:
        chunk_stats = [r for r in chunk_stats if getattr(r, "config_id_used") == config_id_used]
    if not chunk_stats:
        raise ValueError("No rows matched the requested config_id_used")

    # Book-level arrays (each book = one sample)
    book_chunk_counts = np.asarray([getattr(r, "chunk_count") for r in chunk_stats], dtype=np.int64)
    book_token_means  = np.asarray([getattr(r, "token_mean") for r in chunk_stats], dtype=np.float64)
    book_token_stds   = np.asarray([getattr(r, "token_std") for r in chunk_stats], dtype=np.float64)
    book_token_maxs   = np.asarray([getattr(r, "token_max") for r in chunk_stats], dtype=np.int64)

    # Chunk-level pooled array (all chunks across all books)
    pooled = []
    total_chunks = 0
    for r in chunk_stats:
        tc = getattr(r, "token_counts", None)
        if tc is None:
            raise ValueError("Row missing token_counts; cannot compute pooled chunk-level percentiles.")
        pooled.append(np.asarray(tc, dtype=np.int64))
        total_chunks += int(getattr(r, "chunk_count"))

    pooled_chunk_tokens = np.concatenate(pooled) if pooled else np.asarray([], dtype=np.int64)
    if pooled_chunk_tokens.size == 0:
        raise ValueError("No pooled chunk tokens available (empty token_counts across rows).")

    #  Percentiles 
    book_chunk_count_median = float(np.percentile(book_chunk_counts, 50))
    book_chunk_count_p90    = float(np.percentile(book_chunk_counts, 90))

    book_token_mean_median  = float(np.percentile(book_token_means, 50))
    book_token_mean_p90     = float(np.percentile(book_token_means, 90))

    book_token_std_median   = float(np.percentile(book_token_stds, 50))
    book_token_std_p90      = float(np.percentile(book_token_stds, 90))

    book_token_max_median   = float(np.percentile(book_token_maxs, 50))
    book_token_max_p90      = float(np.percentile(book_token_maxs, 90))

    chunk_token_p10         = float(np.percentile(pooled_chunk_tokens, 10))
    chunk_token_p50         = float(np.percentile(pooled_chunk_tokens, 50))
    chunk_token_p90         = float(np.percentile(pooled_chunk_tokens, 90))
    chunk_token_p99         = float(np.percentile(pooled_chunk_tokens, 99))

    # Outlier rates
    p90_token_std = book_token_std_p90
    p90_token_max = book_token_max_p90
    p99_chunk_count = float(np.percentile(book_chunk_counts, 99))

    pct_std_gt_p90 = float(np.mean(book_token_stds > p90_token_std) * 100.0)
    pct_max_gt_2xp90 = float(np.mean(book_token_maxs > (2.0 * p90_token_max)) * 100.0)
    pct_cc_gt_p99 = float(np.mean(book_chunk_counts > p99_chunk_count) * 100.0)

    return CollectionFingerprint(
                config_id_used=config_id_used,
                book_count=int(book_chunk_counts.size),
                total_chunks=int(total_chunks),

                book_chunk_count_median=book_chunk_count_median,
                book_chunk_count_p90=book_chunk_count_p90,

                book_token_mean_median=book_token_mean_median,
                book_token_mean_p90=book_token_mean_p90,

                book_token_std_median=book_token_std_median,
                book_token_std_p90=book_token_std_p90,

                book_token_max_median=book_token_max_median,
                book_token_max_p90=book_token_max_p90,

                chunk_token_p10=chunk_token_p10,
                chunk_token_p50=chunk_token_p50,
                chunk_token_p90=chunk_token_p90,
                chunk_token_p99=chunk_token_p99,

                pct_books_token_std_gt_p90=pct_std_gt_p90,
                pct_books_token_max_gt_2xp90=pct_max_gt_2xp90,
                pct_books_chunk_count_gt_p99=pct_cc_gt_p99,
            )


