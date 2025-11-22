from __future__ import annotations
import csv, asyncio
from pathlib import Path
from azure_judge import AzureJudgeModel
from settings import get_settings
from retrieval.retrieve import answer_rag

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.models import AzureOpenAIModel

TEST_PARAMS = {
    "top_n": 10
}


# RAG PIPELINE HOOK
async def run_gutenberg_rag(question: str) -> tuple[str, list[str]]:
    """
    Entire RAG retrival hook
    Returns:
        - answer: str                   (model's final answer)
        - contexts: list[str]           (list of retrieved passages / chunks)
    """
    sett = get_settings()
    q_resp = await answer_rag(query=question, sett=sett, top_n_matches=TEST_PARAMS["top_n"])
    
    contexts_found = [c.content for c in q_resp.citations if c.content]
    return q_resp.answer, contexts_found

def load_golden_dataset(csv_path: Path) -> list[dict]:
    """
    Expects CSV with columns:
        gb_id,book,author,question,ground_truth,chapter_section,gold_passage_hint,difficulty,type
    """
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

async def build_test_cases(golden_rows: list[dict]) -> list[LLMTestCase]:
    test_cases: list[LLMTestCase] = []

    for row in golden_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]

        # Call your RAG system
        answer, retrieval_context = await run_gutenberg_rag(question)

        # TODO: add metadata for the answer?
        # Some useful metadata you might want to see in reports
        metadata = {
            "gb_id": row.get("gb_id"),
            "book": row.get("book"),
            "author": row.get("author"),
            "chapter_section": row.get("chapter_section"),
            "difficulty": row.get("difficulty"),
            "type": row.get("type"),
            "gold_passage_hint": row.get("gold_passage_hint"),
        }

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,   # used by some metrics
            retrieval_context=retrieval_context,
            additional_metadata=metadata,
        )
        test_cases.append(test_case)

    return test_cases


#-- DEFINE METRICS & RUN EVALUATION --

async def main(csv_path: Path) -> None:
    golden_rows = load_golden_dataset(csv_path)
    test_cases = await build_test_cases(golden_rows)

    sett = get_settings()

    # az_model = AzureOpenAIModel(
    #                 model_name="gpt-5-mini",
    #                 deployment_name=sett.AZ_OPENAI_MODEL_DEPLOYMENT,
    #                 azure_openai_api_key=sett.AZ_OPENAI_GPT_KEY,
    #                 openai_api_version=sett.AZ_OPENAI_API_VER,
    #                 azure_endpoint=sett.AZ_OPENAI_GPT_ENDPOINT,
    #                 # temperature=0
    #             )

    az_model_judge = AzureJudgeModel(
                            azure_endpoint="https://moby-rag-ai-foundry.cognitiveservices.azure.com",#sett.AZ_OPENAI_GPT_ENDPOINT,
                            api_key=sett.AZ_OPENAI_GPT_KEY,
                            api_version=sett.AZ_OPENAI_API_VER,
                            deployment_name=sett.AZ_OPENAI_MODEL_DEPLOYMENT,
                            model_name="gpt-5-mini",
                            # temperature=0.0,
                        )

    # RAG metrics:
    # - AnswerRelevancy: how well your answer responds to the question
    # - Faithfulness: does answer stick to retrieved context (hallucinations?)
    # - ContextualPrecision / Recall: how good your retrieval is
     # Optional: quick sanity check to confirm DeepEval sees your model
    print("Eval model:", az_model_judge.get_model_name())

    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=az_model_judge)       # generator metric
    faithfulness = FaithfulnessMetric(threshold=0.7, model=az_model_judge)              # generator metric
    contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=az_model_judge)  # retriever metric
    contextual_recall = ContextualRecallMetric(threshold=0.7, model=az_model_judge)        # retriever metric

    metrics = [
        answer_relevancy,
        faithfulness,
        contextual_precision,
        contextual_recall,
    ]

    print(f"Loaded {len(test_cases)} test cases from {csv_path}")
    print("Running DeepEval RAG evaluation...\n")

    # This will run all metrics over all test cases
    evaluate(test_cases, metrics=metrics, )


if __name__ == "__main__":
    asyncio.run(main(Path("eval_data", "gutenberg_gold_small.csv")))
