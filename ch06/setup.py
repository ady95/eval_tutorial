"""06장 RAGAS 연결 설정: 평가용 LLM(Judge)과 임베딩을 RAGAS 형식으로 만듭니다.

.env 에서 읽는 값은 01-2, 02-2 와 같습니다. 추가로 다음 값을 쓸 수 있습니다.
  OPENAI_DROP_PARAMS  (선택) 서버가 받지 않는 파라미터를 쉼표로 적으면 요청에서 뺍니다.
                      예) temperature,top_p   — OpenAI 공식 API에서는 비워 둡니다.
"""
import os

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory

from common.llm import EMBEDDING_MODEL, JUDGE_MODEL


def drop_params(client: AsyncOpenAI, names: list[str]) -> AsyncOpenAI:
    """chat.completions.create 를 감싸, 지정한 파라미터를 빼고 보냅니다."""
    create = client.chat.completions.create

    async def create_without(*args, **kwargs):
        for name in names:
            kwargs.pop(name, None)
        return await create(*args, **kwargs)

    client.chat.completions.create = create_without
    return client


def ragas_llm():
    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    names = [n.strip() for n in os.getenv("OPENAI_DROP_PARAMS", "").split(",") if n.strip()]
    if names:
        drop_params(client, names)
    return llm_factory(JUDGE_MODEL, client=client)


def ragas_embeddings():
    client = AsyncOpenAI(
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    return embedding_factory("openai", model=EMBEDDING_MODEL, client=client)
