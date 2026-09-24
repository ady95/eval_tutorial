"""07-1 실습: 05장 RAG 결과 세 개를 DeepEval 의 기본 RAG 지표 네 개로 채점합니다.

실행: python -m ch07.first_eval      (먼저 python -m ch05.evaluate_rag 실행)
"""
import json
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ch07.judge_model import OpenAICompatibleJudge

rows = json.loads((Path("results") / "ch05_evaluate_rag_c400_k3.json").read_text(encoding="utf-8"))
samples = [r for r in rows if r["id"] in (12, 19, 29)]

# 05장 결과 한 문항 = LLMTestCase 하나
test_cases = [
    LLMTestCase(
        input=r["question"],
        actual_output=r["answer"],
        expected_output=r["reference"],
        retrieval_context=[c["text"] for c in r["retrieved"]],
        name=f"q{r['id']}",
    )
    for r in samples
]

judge = OpenAICompatibleJudge()
metrics = [
    AnswerRelevancyMetric(model=judge, threshold=0.7),
    FaithfulnessMetric(model=judge, threshold=0.7),
    ContextualPrecisionMetric(model=judge, threshold=0.7),
    ContextualRecallMetric(model=judge, threshold=0.7),
]

result = evaluate(
    test_cases,
    metrics,
    async_config=AsyncConfig(max_concurrent=4),  # 기본값 20은 서버에 한꺼번에 너무 많이 보냅니다
    display_config=DisplayConfig(print_results=False, inspect_after_run=False),
)

for tr in result.test_results:
    print(f"[{tr.name}] {'PASS' if tr.success else 'FAIL'}")
    for md in tr.metrics_data:
        print(f"  {md.name:<22} {md.score:.2f}  {'통과' if md.success else '미달'}  {md.reason[:70]}")
