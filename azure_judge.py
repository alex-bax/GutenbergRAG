from openai import AzureOpenAI
from deepeval.models import DeepEvalBaseLLM


class AzureJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        model_name: str = "azure-judge",
        
    ) -> None:
        # IMPORTANT: azure_endpoint should be the base resource URL, e.g.
        # "https://<resource-name>.openai.azure.com"
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self._model_name = model_name

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> AzureOpenAI:
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Simple string-in / string-out interface.
        We use the Responses API with `input`, and we do NOT accept `schema`.
        """
        client = self.load_model()
        resp = client.responses.create(
            model=self.deployment_name,
            input=prompt,
            # no `response_format`, no `messages`
        )
        return resp.output_text  # type: ignore[attr-defined]

    async def a_generate(self, prompt: str) -> str:
        # DeepEval uses this in async_mode=True.
        # Reuse sync for simplicity.
        return self.generate(prompt)
