"""07장 DeepEval에 우리 Judge 모델을 연결합니다.

DeepEval 의 기본 OpenAI 모델(GPTModel)은 목록에 없는 모델에 temperature=0 을 보내는데,
추론 모델이나 일부 호환 서버는 이를 거부합니다. 01-2 의 클라이언트를 그대로 쓰는 모델 클래스를 만들어
DeepEval 지표의 model 인자로 넘깁니다. 04-4 의 로컬 Judge 도 같은 방식으로 연결할 수 있습니다.
"""
import asyncio

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

from common.llm import JUDGE_MODEL, client


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = JUDGE_MODEL, judge_client: OpenAI = client, **options):
        self.model_name = model
        self.judge_client = judge_client
        self.options = options  # 예: 로컬 Ollama 모델이면 reasoning_effort="none"
        super().__init__(model)

    def load_model(self):
        return self.judge_client

    def generate(self, prompt: str, schema=None) -> str:
        extra = {"response_format": {"type": "json_object"}} if schema else {}
        response = self.judge_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **extra,
            **self.options,
        )
        return response.choices[0].message.content  # JSON 문자열은 DeepEval 이 스키마에 맞춰 읽습니다

    async def a_generate(self, prompt: str, schema=None) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name
