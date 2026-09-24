"""07-2 실습: 누리솔 답변 정책 지표(G-Eval + rubric)를 성격이 다른 답변 다섯 개로 확인합니다.

실행: python -m ch07.geval_policy      (먼저 python -m ch07.deepeval_rag 실행)
"""
import json
from pathlib import Path

from deepeval.test_case import LLMTestCase

from ch07.geval_metrics import policy

rows = {r["id"]: r for r in json.loads((Path("results") / "ch07_deepeval_c400_k3.json").read_text(encoding="utf-8"))}
busan = [c["text"] for c in rows[12]["retrieved"]]  # 현재 국내 출장비 규정 + 폐지된 2025년판
bonus = [c["text"] for c in rows[40]["retrieved"]]  # 명절 상여금 내용이 없는 검색 결과

CASES = [
    ("규정대로 답함", "부산 출장 숙박비는 1박에 얼마까지 나와요?", "부산은 1박당 12만 원까지 실비로 지급됩니다.", busan),
    ("폐지 규정 인용", "부산 출장 숙박비는 1박에 얼마까지 나와요?", "부산은 그 밖의 지역이라 1박당 8만 원까지 지급됩니다.", busan),
    ("단정·보장 표현", "부산 출장 숙박비는 1박에 얼마까지 나와요?",
     "부산은 1박당 12만 원까지 무조건 100% 지급되니 걱정하지 마세요.", busan),
    ("없다고 답함", "명절 상여금은 얼마인가요?", "규정에서 찾을 수 없습니다.", bonus),
    ("추측으로 답함", "명절 상여금은 얼마인가요?", "보통 기본급의 50% 정도가 명절 상여금으로 지급됩니다.", bonus),
]

for name, question, answer, contexts in CASES:
    metric = policy(async_mode=False)
    metric.measure(LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts))
    print(f"[{name}] {metric.score:.1f} {'PASS' if metric.success else 'FAIL'}  {answer}\n  └ {metric.reason}\n")
