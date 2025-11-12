from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas import evaluate
import pandas as pd
from pathlib import Path
from retrieve import search_chunks, answer_with_context
from settings import get_settings

# TODO: calculate cost pr run in tokens
# TODO: take all books and check if in index, if not then upload them

def are_test_books_in_index(books:list[str]) -> :
    is_book_in_index(search_client=search_client, book_id=gb_meta.id)

if __name__ == "__main__":
    eval_dataset_p = Path("eval_data", "gutenberg_gold.csv")
    df = pd.read_csv(eval_dataset_p)

    sett = get_settings()
    records = []

    for i, row in enumerate(df.itertuples(), 1):
        print(i, row.question)

        chunks_found = search_chunks(query=str(row.question), 
                                     search_client=sett.get_search_client(), 
                                     embed_client=sett.get_emb_client(), 
                                     embed_model_deployed=sett.EMBED_MODEL_DEPLOYMENT)

        ans = answer_with_context(query=str(row.question), 
                                  llm_client=sett.get_llm_client(), 
                                  llm_model_deployed=sett.LLM_MODEL_DEPLOYMENT, 
                                  chunk_hits=chunks_found)
        records.append({
            "question": row.question,
            "contexts": chunks_found,
            "answer": ans,
            "ground_truth": row.get("ground_truth", None)   # type:ignore
        })

    ds = Dataset.from_list(records)