import json
import pytest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
from retrieval.retrieve import run_gutenberg_rag
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.models import AzureOpenAIModel
from settings import get_settings, Settings
from typing import AsyncIterator
import pytest_asyncio
from datetime import datetime
from constants import DEF_BOOK_NAMES_TO_IDS

@pytest_asyncio.fixture(scope="session")
async def settings() -> AsyncIterator[Settings]:
    sett = get_settings()
    
    try:
        yield sett
    finally:
        # ensure Qdrant client (and other async resources) are closed
        await sett.close_vector_store()

dataset = EvaluationDataset()

dataset_p = Path("eval_data", "gb_gold.csv")
dataset.add_goldens_from_csv_file(
    file_path=str(dataset_p),
    input_col_name="question",
    name_key_name="book"
)

now = datetime.now().strftime("%H%M_%d%m")
deep_eval_log_p = Path("deep_eval_logs")
deep_eval_log_p.mkdir(exist_ok=True)
deep_eval_log_p = deep_eval_log_p /  f'{dataset_p.stem}_{now}.txt'

def log_metric_outp(metrics:list[BaseMetric], 
                    gold_inp_q:str, 
                    gold_exp_outp:str, 
                    model_ans:str, 
                    contexts:list[str],
                    test_case:LLMTestCase) -> None:
    # metrics_d = []
    d = {"Q":gold_inp_q, "ExpA":gold_exp_outp, 
            "A":model_ans, "Contexts":contexts,
            "Metrics":[]}
    
    
    for m in metrics:
        d["Metrics"].append(
            {"M":m.__class__.__name__,  "Score": m.measure(test_case),  "Threshold": m.threshold, "R":m.reason}
        )
    
    with open(deep_eval_log_p.with_suffix(".json"), 'a') as f:
        json.dump(d, f)


@pytest.mark.asyncio  
@pytest.mark.parametrize("golden", dataset.goldens)
async def test_gutenberg_rag_answer_relevancy(golden: Golden, settings:Settings):
    # TODO: print all books available in vec collection
    vec_store = await settings.get_vector_store()
    books_in_collection = await vec_store.get_all_unique_book_names()
    assert all(book_name in books_in_collection for book_name in DEF_BOOK_NAMES_TO_IDS.keys())

    answer, contexts = await run_gutenberg_rag(golden.input, settings)
    az_model = AzureOpenAIModel(
                model_name="gpt-5-mini",
                deployment_name="gpt-5-mini",
                azure_openai_api_key=settings.AZ_OPENAI_GPT_KEY,
                openai_api_version="2025-04-01-preview",
                azure_endpoint="https://moby-rag-ai-foundry.cognitiveservices.azure.com",
                temperature=1.0
            )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=golden.expected_output,
    )

    answer_rel_metric = AnswerRelevancyMetric(threshold=0.7, model=az_model)
    faith_metric = FaithfulnessMetric(threshold=0.7, model=az_model)
    context_rel_metric = ContextualRelevancyMetric(threshold=0.7, model=az_model)
    context_prec_metric = ContextualPrecisionMetric(threshold=0.7, model=az_model)
    metrics = [answer_rel_metric, faith_metric,
                context_prec_metric, context_rel_metric]
    
    log_metric_outp(metrics=metrics, 
                    gold_inp_q=golden.input, 
                    gold_exp_outp=golden.expected_output if golden.expected_output else "", 
                    model_ans=answer,
                    contexts=contexts,
                    test_case=test_case,
                )
    
    try:
        assert_test(test_case=test_case,
                    metrics=metrics)
    except Exception as ex:
        print("FAILED golden:", golden.input, golden.name)
        print("Answer:", answer)
        print("Contexts:", contexts)
        raise

# @deepeval.log_hyperparameters(model="gpt-5-mini", prompt_template="...")
# def hyperparameters():
#     return {"model": "gpt-5-mini", "system prompt": "..."}

