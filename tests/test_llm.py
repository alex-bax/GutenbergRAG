import pytest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
# from your_agent import your_llm_app # Replace with your LLM app
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

# @pytest.fixture(scope="session")
@pytest_asyncio.fixture(scope="session")
async def settings() -> AsyncIterator[Settings]:
    sett = get_settings()
    
    try:
        yield sett
    finally:
        # ensure Qdrant client (and other async resources) are closed
        await sett.close_vector_store()

dataset = EvaluationDataset()

dataset_p = Path("eval_data", "gutenberg_gold_small.csv")
dataset.add_goldens_from_csv_file(
    file_path=str(dataset_p),
    input_col_name="question"
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
  with open(deep_eval_log_p, "a", encoding="utf-8") as f:
    for m in metrics:
        f.write("\n=== DeepEval Detailed Case ===")
        f.write(f"Q: {gold_inp_q}\n")
        f.write(f"Exp: {gold_exp_outp}\n")
        f.write(f"A:   {model_ans}\n")
        f.write(f"Context ({len(contexts)} chunks):\n")
        for i, c in enumerate(contexts):
            f.write(f"  [{i}] {c}\n")
        f.write(f"Metric: {m.__class__.__name__}  Score: {m.measure(test_case)}  Threshold: {m.threshold}\n")
        f.write(f"R:{m.reason}")
        f.write("================================\n")


@pytest.mark.asyncio  
@pytest.mark.parametrize("golden", dataset.goldens)
async def test_gutenberg_rag_answer_relevancy(golden: Golden, settings:Settings):
    # TODO: print all books available in vec collection


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

    assert_test(
        test_case=test_case,
        metrics=metrics
    )

# @deepeval.log_hyperparameters(model="gpt-5-mini", prompt_template="...")
# def hyperparameters():
#     return {"model": "gpt-5-mini", "system prompt": "..."}

