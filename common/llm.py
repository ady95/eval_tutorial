"""책 전체에서 함께 쓰는 LLM 호출 도구.

.env 에서 다음 값을 읽습니다.
  OPENAI_API_KEY      API 키
  OPENAI_BASE_URL     (선택) OpenAI 호환 서버 주소. 비우면 OpenAI 공식 API
  OPENAI_MODEL        답변을 만드는 모델 (평가 대상)
  OPENAI_MODEL_JUDGE  답변을 채점하는 모델 (Judge). 비우면 OPENAI_MODEL 과 같음
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OPENAI_BASE_URL 이 빈 값("")이면 openai 라이브러리가 공식 주소로 바꿔 주지 않으므로 직접 지정합니다
client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
MODEL = os.environ["OPENAI_MODEL"]
JUDGE_MODEL = os.getenv("OPENAI_MODEL_JUDGE") or MODEL


def chat(prompt: str, model: str = MODEL, **kwargs) -> str:
    """질문 하나를 보내고 답변 문자열을 돌려줍니다."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


# 임베딩은 대화 모델과 다른 서버(예: Ollama의 bge-m3)를 쓸 수 있도록 설정을 따로 둡니다
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "bge-m3"
embed_client = OpenAI(
    base_url=os.getenv("EMBEDDING_BASE_URL") or str(client.base_url),
    api_key=os.getenv("EMBEDDING_API_KEY") or client.api_key,
)


def embed(texts: list[str]) -> list[list[float]]:
    """문장 목록을 임베딩 벡터 목록으로 바꿉니다."""
    response = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
