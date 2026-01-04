from prometheus_client import Histogram

rag_stage_seconds = Histogram(
    "rag_stage_seconds",
    "Time spent in RAG pipeline stages",
    labelnames=("stage",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21)      # after 21 the buckets scale upwards
)
