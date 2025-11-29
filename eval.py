from datasets import Dataset
from tqdm import tqdm
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from datetime import datetime
from pathlib import Path
import pandas as pd
import asyncio
from embedding_pipeline import create_embeddings_async
from settings import get_settings, Settings
from retrieval.retrieve import answer_with_context
from ingestion.book_loader import upload_missing_book_ids
# TODO: calculate cost pr run in tokens
# TODO: take all books and check if in index, if not then upload them

def get_ragas_wrapped_llm_and_embeddings(sett:Settings):
    ragas_llm = llm_factory(model="gpt-5-mini", provide="openai", client=sett.get_llm_client())
    ragas_emb = embedding_factory(model="text-embedding-3-small", client=sett.get_emb_client())

    return ragas_llm, ragas_emb


async def build_eval_dataset():
    eval_dataset_p = Path("eval_data", "gutenberg_gold_small.csv")
    df = pd.read_csv(eval_dataset_p)

    sett = get_settings()
    records = []

    vector_store = await sett.get_vector_store()

    ## Ingestion
    test_book_ids = set(list(df["gb_id"].unique().tolist())[:2])

    # TODO: update to use test sess, i.e. not prod DB and vector store
    gb_books = await upload_missing_book_ids(book_ids=test_book_ids, sett=sett, db_sess=)

    ## Retrival
    req_lim, tok_lim = sett.get_limiters()

    for i, row in tqdm(enumerate(df.itertuples(), 1)):
        print(f" {i} - {row.question}")

        emb_client = sett.get_async_emb_client() 
        emb_vecs = await create_embeddings_async(embed_client=emb_client, 
                                                model_deployed=sett.EMBED_MODEL_DEPLOYMENT,
                                                inp_batches=[[str(row.question)]],
                                                req_limiter=req_lim,
                                                tok_limiter=tok_lim
                                                ) 
        
        chunks_found = await vector_store.search_by_embedding(embed_query_vector=emb_vecs[0], filter=None)

        ans, relevant_chunks = answer_with_context(query=str(row.question), 
                                                    llm_client=sett.get_llm_client(), 
                                                    llm_model_deployed=sett.AZ_OPENAI_MODEL_DEPLOYMENT, 
                                                    chunk_hits=chunks_found)
        records.append({
            "question": row.question,
            "contexts": [c.content for c in chunks_found],
            "answer": ans,
            "ground_truth": row.ground_truth,

            # Extra metadata
            "gb_id": row.gb_id,
            "book": str(row.book),
            "author": str(row.author),
            "chapter_section": str(row.chapter_section),
            "gold_passage_hint": str(row.gold_passage_hint),
            "difficulty": str(row.difficulty),
            "q_type": str(row.type),

            # Retrieval metadata
            "retrieved_book_ids": [c.book_id for c in chunks_found],
            "retrieved_chunk_nrs": [c.chunk_nr for c in chunks_found],
        })

    ds = Dataset.from_list(records)
    ds.to_csv(Path("eval_data", "results", f"eval_{datetime.now().strftime("%H%M-%d%m")}.csv"))

    return ds

def run_ragas_eval(ds: Dataset):
    sett = get_settings()
    ragas_llm, ragas_emb = get_ragas_wrapped_llm_and_embeddings(sett)

    result = evaluate(
        ds,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        return_executor=False
    )

    # Save full results (including per-row metric columns)
    res_out_dir = Path("eval_data", "results")
    res_out_dir.mkdir(parents=True, exist_ok=True)
    df_metrics = result.to_pandas() # type: ignore

    
    print(result)
    # print(f'Total cost: {result.total_cost()}') 

    print(df_metrics.describe())
    df_metrics.to_csv(Path(res_out_dir) / Path("ragas_res.csv"), index=False)
    return result

async def main():
    ds = await build_eval_dataset()
    ragas_res = run_ragas_eval(ds)
    

if __name__ == "__main__":
    asyncio.run(main())