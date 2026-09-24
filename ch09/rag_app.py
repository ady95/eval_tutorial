"""09장 계측(instrumentation)을 붙인 누리솔 RAG: 답변과 함께 단계별 시간과 토큰 수를 기록합니다.

05장의 RAG 와 같은 검색기·프롬프트를 쓰고, 설정(청크 크기, top_k, Reranker, 프롬프트, 모델)을 바꿔 끼울 수 있게 했습니다.
"""
import time
from dataclasses import asdict, dataclass

from openai import OpenAI

from ch05.rag import SYSTEM, Retriever
from common.llm import MODEL, client

PROMPTS = {
    "v1": SYSTEM,  # 05장 프롬프트
    # 07-3 에서 테스트를 깨뜨렸던 프롬프트: "찾을 수 없다는 답이 많다"는 불만을 듣고 고쳤다고 가정합니다
    "friendly": """당신은 친절한 누리솔 사내 도우미입니다.
아래 [참고 문서]를 참고해 질문에 답하세요.
"찾을 수 없다"는 답은 직원에게 도움이 되지 않으니 하지 마세요. 문서에 없는 내용은 일반적인 회사 기준으로 안내하세요.
답변은 두세 문장으로 간결하게 쓰세요.""",
}

# CI 에서 매번 돌리는 작은 평가셋(스모크 테스트): 유형별로 골랐고, 09-2 의 기준선 3회에서 모두 PASS 한 문항입니다
SMOKE_IDS = [1, 4, 10, 17, 22, 29, 31, 35, 37, 40]


@dataclass
class RagConfig:
    chunk_size: int = 400
    top_k: int = 3
    rerank: bool = False     # True 면 10개를 뽑아 06-6 의 Reranker 로 top_k 개를 고릅니다
    prompt: str = "v1"
    model: str = MODEL
    base_url: str = ""       # 비우면 common.llm 의 클라이언트. 로컬 Ollama 면 http://localhost:11434/v1

    def name(self) -> str:
        parts = [f"c{self.chunk_size}", f"k{self.top_k}", "rr" * self.rerank, self.prompt, self.model.replace(":", "_")]
        return "_".join(p for p in parts if p)


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class RagApp:
    def __init__(self, config: RagConfig):
        self.config = config
        self.retriever = Retriever(config.chunk_size)
        self.client = OpenAI(base_url=config.base_url, api_key="ollama") if config.base_url else client
        self.options = {"reasoning_effort": "none"} if config.base_url else {}  # 04-4 처럼 로컬 모델의 추론 모드를 끕니다

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, 10 if self.config.rerank else self.config.top_k)
        search_ms = elapsed_ms(t0)
        t0 = time.perf_counter()
        if self.config.rerank:
            from ch06.rerank import rerank  # Reranker 를 쓸 때만 torch 를 불러옵니다
            hits = rerank(question, hits, self.config.top_k)
        rerank_ms = elapsed_ms(t0)
        contexts = [chunk.text for chunk, _ in hits]
        docs = "\n\n---\n\n".join(contexts)
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": PROMPTS[self.config.prompt]},
                      {"role": "user", "content": f"[참고 문서]\n{docs}\n\n[질문]\n{question}"}],
            **self.options,
        )
        llm_ms = elapsed_ms(t0)
        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "retrieved": [{"doc_id": c.doc_id, "index": c.index, "text": c.text} for c, _ in hits],
            "search_ms": search_ms, "rerank_ms": rerank_ms, "llm_ms": llm_ms,
            "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens,
            "config": asdict(self.config),
        }
