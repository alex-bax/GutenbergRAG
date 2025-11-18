from datasets import Dataset
from tqdm import tqdm
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas import evaluate
import pandas as pd
from pathlib import Path

from azure.search.documents import SearchClient
from search_handler import create_missing_search_index
from settings import get_settings, Settings
from retrieve import search_chunks, answer_with_context
# TODO: calculate cost pr run in tokens
# TODO: take all books and check if in index, if not then upload them
from load_book import index_upload_missing_book_ids
import asyncio
from datetime import datetime

async def run_eval():
    eval_dataset_p = Path("eval_data", "gutenberg_gold.csv")
    df = pd.read_csv(eval_dataset_p)

    sett = get_settings()
    records = []

    ## Ingestion
    test_book_ids = list(df["gb_id"].unique().tolist())[:2]
    create_missing_search_index(search_index_client=sett.get_index_client())

    gb_books = await index_upload_missing_book_ids(book_ids=test_book_ids, sett=sett)

    ## Retrival
    for i, row in tqdm(enumerate(df.itertuples(), 1)):
        print(f" {i} - {row.question}")

        chunks_found = search_chunks(query=str(row.question), 
                                     search_client=sett.get_search_client(), 
                                     embed_client=sett.get_emb_client(), 
                                     embed_model_deployed=sett.EMBED_MODEL_DEPLOYMENT)

        ans, relevant_chunks = answer_with_context(query=str(row.question), 
                                  llm_client=sett.get_llm_client(), 
                                  llm_model_deployed=sett.LLM_MODEL_DEPLOYMENT, 
                                  chunk_hits=chunks_found)
        records.append({
            "question": row.question,
            "contexts": chunks_found,
            "answer": ans,
            "ground_truth": row.ground_truth
        })

    ds = Dataset.from_list(records)
    ds.to_csv(Path("eval_data", "results", f"eval_{datetime.now().strftime("%H%M-%d%m")}.csv"))

if __name__ == "__main__":
    asyncio.run(run_eval())