"""07-3 실습: 누리솔 RAG 를 pytest 테스트로 확인합니다. 테스트할 때마다 답변을 새로 만듭니다.

실행: deepeval test run ch07/test_rag.py
프롬프트를 바꿔 실행: 환경 변수 RAG_PROMPT=friendly 를 주고 같은 명령을 실행합니다
"""
import os

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ch05.rag import SYSTEM, Retriever, load_eval_set
from ch07.geval_metrics import correctness_steps, policy
from common.llm import MODEL, client

PROMPTS = {
    "default": SYSTEM,  # 05장에서 쓴 프롬프트
    # "찾을 수 없다는 답이 많다"는 불만을 듣고 누군가 프롬프트를 고쳤다고 가정합니다
    "friendly": """당신은 친절한 누리솔 사내 도우미입니다.
아래 [참고 문서]를 참고해 질문에 답하세요.
"찾을 수 없다"는 답은 직원에게 도움이 되지 않으니 하지 마세요. 문서에 없는 내용은 일반적인 회사 기준으로 안내하세요.
답변은 두세 문장으로 간결하게 쓰세요.""",
}
PROMPT = PROMPTS[os.getenv("RAG_PROMPT", "default")]

# 회귀 테스트 문항: 07-2 에서 두 번 모두 0.9 이상을 받은 문항 중 유형별로 골랐습니다
REGRESSION_IDS = [1, 17, 29, 35, 37, 40, 41]
CASES = [q for q in load_eval_set() if q["id"] in REGRESSION_IDS]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(400)  # 문서 색인은 테스트 파일마다 한 번만 만듭니다


def generate(question: str, contexts: list[str]) -> str:
    docs = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": f"[참고 문서]\n{docs}\n\n[질문]\n{question}"}],
    )
    return response.choices[0].message.content.strip()


@pytest.mark.parametrize("case", CASES, ids=[f"q{q['id']}" for q in CASES])
def test_rag_answer(case, retriever):
    contexts = [c.text for c, _ in retriever.search(case["question"], 3)]
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=generate(case["question"], contexts),
        expected_output=case["reference"],
        retrieval_context=contexts,
    )
    assert_test(test_case, [correctness_steps(), policy()])
