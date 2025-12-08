import pytest
from pathlib import Path
from deepeval.dataset import EvaluationDataset
# from your_agent import your_llm_app # Replace with your LLM app
from retrieval.retrieve import run_gutenberg_rag
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.models import AzureOpenAIModel
from settings import get_settings, Settings
from typing import AsyncGenerator, AsyncIterator
import pytest_asyncio

# @pytest.fixture(scope="session")
@pytest_asyncio.fixture(scope="session")
async def settings() -> AsyncIterator[Settings]:
    sett = get_settings()
    # sett.is_test = True       
    
    try:
        yield sett
    finally:
        # ensure Qdrant client (and other async resources) are closed
        await sett.close_vector_store()

dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path=str(Path("eval_data", "gutenberg_gold_small.csv")),
    input_col_name="question"
)

@pytest.mark.asyncio  # requires pytest-asyncio installed
@pytest.mark.parametrize("golden", dataset.goldens)
async def test_gutenberg_rag_answer_relevancy(golden: Golden, settings:Settings):
    
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
        # If your CSV has a “ground truth” column, you can wire it here:
        expected_output=golden.expected_output,
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=az_model)

    assert_test(
        test_case=test_case,
        metrics=[metric],
    )
# @deepeval.log_hyperparameters(model="gpt-5-mini", prompt_template="...")
# def hyperparameters():
#     return {"model": "gpt-5-mini", "system prompt": "..."}

