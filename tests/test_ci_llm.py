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
from settings import get_settings

dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path=str(Path("eval_data", "gutenberg_gold_small.csv")),
    input_col_name="question"
)

# Loop through goldens using pytest

# @pytest.mark.parametrize("golden",dataset.goldens)
# async def test_llm_app(golden: Golden):
#     res, text_chunks = await run_gutenberg_rag(golden.input)
#     test_case = LLMTestCase(input=golden.input, 
#                             actual_output=res, 
#                             retrieval_context=text_chunks)
#     assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])


@pytest.mark.asyncio  # requires pytest-asyncio installed
@pytest.mark.parametrize("golden", dataset.goldens)
async def test_gutenberg_rag_answer_relevancy(golden: Golden):
    
    sett = get_settings()
    answer, contexts = await run_gutenberg_rag(golden.input, sett)
    az_model = AzureOpenAIModel(
                model_name="gpt-5-mini",
                deployment_name="gpt-5-mini",
                azure_openai_api_key=sett.AZ_OPENAI_GPT_KEY,
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

    # threshold makes the test fail if metric < threshold
    metric = AnswerRelevancyMetric(threshold=0.7, model=az_model)

    assert_test(
        test_case=test_case,
        metrics=[metric],
    )
# @deepeval.log_hyperparameters(model="gpt-5-mini", prompt_template="...")
# def hyperparameters():
#     return {"model": "gpt-5-mini", "system prompt": "..."}

