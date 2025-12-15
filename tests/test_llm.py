import json, time, pytest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
from config.params import ConfigParamSettings
from evals.timer_helper import Timer
from retrieval.retrieve import run_gutenberg_rag
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval import assert_test, evaluate
from deepeval.models import AzureOpenAIModel
from config.settings import get_settings, Settings
from typing import AsyncIterator
import pytest_asyncio
from datetime import datetime


@pytest_asyncio.fixture(scope="session")
async def settings() -> AsyncIterator[Settings]:
    sett = get_settings(hyperparam_p=Path("config", "hp-ch400.json"))
    
    try:
        yield sett
    finally:
        # ensure Qdrant client (and other async resources) are closed
        await sett.close_vector_store()

dataset = EvaluationDataset()
dataset_p = Path("evals", "datasets", "gb_gold_med.csv")

dataset.add_goldens_from_csv_file(
    file_path=str(dataset_p),
    input_col_name="question",
    name_key_name="book"
)

now = datetime.now().strftime("%H%M_%d%m-%Y")
data_file_stem = f'{dataset_p.stem}_{now}'
eval_log_dir = Path("evals", now)
eval_log_dir.mkdir(exist_ok=True)
eval_log_p = eval_log_dir / f'{data_file_stem}.txt'

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


def log_hyperparams(config:ConfigParamSettings) -> None:
    p = Path("evals", now, f'hp_{data_file_stem}.json')
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, 'w', encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=4)
        

def log_metric_outp(metrics:list[BaseMetric], 
                    gold_inp_q:str, 
                    gold_exp_outp:str, 
                    model_ans:str, 
                    contexts:list[str],
                    test_case:LLMTestCase) -> None:
    d = {"Q":gold_inp_q, "ExpA":gold_exp_outp, 
            "A":model_ans, "Contexts":contexts,
            "Metrics":[]}
    
    for m in metrics:
        d["Metrics"].append(
            {"M":m.__class__.__name__,  
             "Score": m.measure(test_case),  
             "Threshold": m.threshold, "R":m.reason}
        )
    
    with open(eval_log_p.with_suffix(".json"), 'a') as f:
        json.dump(d, f, indent=4)


@pytest.fixture(scope="session")
def deepeval_az_model(settings: "Settings") -> AzureOpenAIModel:
    return AzureOpenAIModel(
        model_name="gpt-5-nano",#"gpt-5-mini",
        deployment_name="gpt-5-nano",#"gpt-5-mini",
        azure_openai_api_key=settings.AZ_OPENAI_GPT_KEY,
        openai_api_version="2025-04-01-preview",
        azure_endpoint="https://moby-rag-ai-foundry.cognitiveservices.azure.com",
        temperature=1.0,
    )

# @pytest.fixture(scope="session")
# def timer() -> Timer:
#     return Timer(enabled=True)


@pytest.mark.asyncio  
@pytest.mark.parametrize("test_case", dataset.test_cases)
async def test_gutenberg_rag_answer_relevancy(test_case:LLMTestCase,#golden: Golden, 
                                              settings:Settings, 
                                              deepeval_az_model:AzureOpenAIModel):
    
    t = Timer(out_path=Path(eval_log_dir, "timings.json"), enabled=True)
    with t.start_timer(key="init_vector_store"):
        vec_store = await settings.get_vector_store()
        books_in_collection = await vec_store.get_all_unique_book_names()
        hp_ing = settings.get_hyperparams().ingestion
        assert all(book_name in books_in_collection for book_name in hp_ing.default_ids_used.keys())

    answer, contexts = await run_gutenberg_rag(test_case.input, settings, timer=t)
    test_case.actual_output = answer
    test_case.retrieval_context = contexts

    ans_rel_met = AnswerRelevancyMetric(threshold=0.7, model=deepeval_az_model)
    faith_met = FaithfulnessMetric(threshold=0.7, model=deepeval_az_model)
    context_rel_metric = ContextualRelevancyMetric(threshold=0.675, model=deepeval_az_model)
    context_prec_metric = ContextualPrecisionMetric(threshold=0.675, model=deepeval_az_model)
    metrics = [ans_rel_met, faith_met,
                context_prec_metric, context_rel_metric]
    # metrics:list[BaseMetric] = [context_rel_metric]#, context_prec_metric]  
    
    log_hyperparams(config=settings.get_hyperparams())
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
    
    except Exception as ex:
        print("FAILED golden:", test_case.input, test_case.name)
        print("Answer:", answer)
        print("Contexts:", contexts)

    t.save()

