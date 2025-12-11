import json
import time
import pytest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
from retrieval.retrieve import run_gutenberg_rag
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval import assert_test, evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.evaluate import DisplayConfig
from config.settings import get_settings, Settings
from typing import AsyncIterator
import pytest_asyncio
from datetime import datetime
from config.hyperparams import DEF_BOOK_NAMES_TO_IDS
@pytest_asyncio.fixture(scope="session")
async def settings() -> AsyncIterator[Settings]:
    sett = get_settings()
    
    try:
        yield sett
    finally:
        # ensure Qdrant client (and other async resources) are closed
        await sett.close_vector_store()

dataset = EvaluationDataset()

# dataset_p = Path("eval_data", "gb_gold.csv")
dataset_p = Path("eval_data", "gb_gold_med.csv")
dataset.add_goldens_from_csv_file(
    file_path=str(dataset_p),
    input_col_name="question",
    name_key_name="book"
)

now = datetime.now().strftime("%H%M_%d%m")
eval_log_p = Path("eval_logs")
eval_log_p.mkdir(exist_ok=True)
eval_log_p = eval_log_p /  f'{dataset_p.stem}_{now}.txt'

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
    
    with open(eval_log_p.with_suffix(".json"), 'a') as f:
        json.dump(d, f)

for golden in dataset.goldens:
    assert isinstance(golden, Golden)
    test_case = LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            additional_metadata={
                "golden_name": getattr(golden, "name", None),
            },
        )
    dataset.add_test_case(test_case)

#TODO: consider using nano
@pytest.fixture(scope="session")
def deepeval_az_model(settings: "Settings") -> AzureOpenAIModel:
    return AzureOpenAIModel(
        model_name="gpt-5-mini",
        deployment_name="gpt-5-mini",
        azure_openai_api_key=settings.AZ_OPENAI_GPT_KEY,
        openai_api_version="2025-04-01-preview",
        azure_endpoint="https://moby-rag-ai-foundry.cognitiveservices.azure.com",
        temperature=1.0,
    )


@pytest.mark.asyncio  
@pytest.mark.parametrize("test_case", dataset.test_cases)
async def test_gutenberg_rag_answer_relevancy(test_case:LLMTestCase,#golden: Golden, 
                                              settings:Settings, 
                                              deepeval_az_model:AzureOpenAIModel):
    
    t0 = time.perf_counter()
    vec_store = await settings.get_vector_store()
    books_in_collection = await vec_store.get_all_unique_book_names()
    assert all(book_name in books_in_collection for book_name in DEF_BOOK_NAMES_TO_IDS.keys())

    answer, contexts = await run_gutenberg_rag(test_case.input, settings)
    t_rag = time.perf_counter()
    print(f"->->->-> TOTAL RAG TIME:{t_rag - t0}")

    test_case.actual_output = answer
    test_case.retrieval_context = contexts
    

    # answer_rel_metric = AnswerRelevancyMetric(threshold=0.7, model=az_model)
    # faith_metric = FaithfulnessMetric(threshold=0.7, model=az_model)
    context_rel_metric = ContextualRelevancyMetric(threshold=0.7, model=deepeval_az_model)
    context_prec_metric = ContextualPrecisionMetric(threshold=0.7, model=deepeval_az_model)
    # metrics = [answer_rel_metric, faith_metric,
    #             context_prec_metric, context_rel_metric]
    metrics:list[BaseMetric] = [context_rel_metric, context_prec_metric]  
    
    log_metric_outp(metrics=metrics, # type:ignore
                    gold_inp_q=test_case.input, 
                    gold_exp_outp=test_case.expected_output if test_case.expected_output else "", 
                    model_ans=answer,
                    contexts=contexts,
                    test_case=test_case,
                )
    try:
        assert_test(test_case=test_case,
                    metrics=metrics,
                    run_async=True,
                    )
        # result = evaluate(
        #     [test_case],
        #     metrics,
        #     display_config=DisplayConfig(
        #         print_results=True,   # print per-metric scores
        #         verbose_mode=True
        #     )
        # )
        # for tr in result.test_results:
        #     if tr.metrics_data:
        #         for mr in tr.metrics_data:
        #             if mr.score:
        #                 assert mr.score >= mr.threshold
    except Exception as ex:
        print("FAILED golden:", test_case.input, test_case.name)
        print("Answer:", answer)
        print("Contexts:", contexts)
        # raise
        print(
            f"[{test_case.input[:7]!r}] "
            f"RAG: {t_rag - t0:.2f}s, "
        )

    t_eval = time.perf_counter()
    print(
        f"[{test_case.input[:7]!r}] "
        f"RAG: {t_rag - t0:.2f}s, "
        f"Metrics: {t_eval - t_rag:.2f}s, "
        f"Total: {t_eval - t0:.2f}s"
    )

# @deepeval.log_hyperparameters(model="gpt-5-mini", prompt_template="...")
# def hyperparameters():
#     return {"model": "gpt-5-mini", "system prompt": "..."}

