"""07-1 실습: 05장 RAG 결과 42문항을 DeepEval 지표로 채점하고 06장의 RAGAS 결과와 비교합니다.

실행: python -m ch07.deepeval_rag      (먼저 python -m ch06.ragas_eval 실행)
"""
import json
import os
import time
from pathlib import Path

# evaluate() 전체를 기다리는 시간 한도(기본 약 207초)를 넉넉히 늘립니다.
# 한도를 넘기면 끝나지 않은 문항이 ignore_errors=True 아래에서 조용히 결과에서 빠집니다.
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS", "1800")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from ch04.agreement import spearman
from ch07.judge_model import OpenAICompatibleJudge

rows = json.loads((Path("results") / "ch06_ragas_c400_k3.json").read_text(encoding="utf-8"))
judge = OpenAICompatibleJudge()

METRICS = {
    "Answer Relevancy": AnswerRelevancyMetric(model=judge, threshold=0.7),
    "Faithfulness": FaithfulnessMetric(model=judge, threshold=0.7),
    "Faithfulness (strict)": FaithfulnessMetric(model=judge, threshold=0.7, penalize_ambiguous_claims=True),
    "Contextual Precision": ContextualPrecisionMetric(model=judge, threshold=0.7),
    "Contextual Recall": ContextualRecallMetric(model=judge, threshold=0.7),
}
# DeepEval 지표 → 06장 RAGAS 지표 (비교용)
RAGAS_PAIR = {
    "Answer Relevancy": "answer_relevancy",
    "Faithfulness": "faithfulness",
    "Faithfulness (strict)": "faithfulness",
    "Contextual Precision": "context_precision",
    "Contextual Recall": "context_recall",
}

test_cases = [
    LLMTestCase(input=r["question"], actual_output=r["answer"], expected_output=r["reference"],
                retrieval_context=[c["text"] for c in r["retrieved"]], name=str(r["id"]))
    for r in rows
]

start = time.time()
result = evaluate(
    test_cases,
    list(METRICS.values()),
    async_config=AsyncConfig(max_concurrent=4),
    display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
    error_config=ErrorConfig(ignore_errors=True),  # 한 문항의 오류로 전체가 멈추지 않게 합니다
    cache_config=CacheConfig(write_cache=False),  # 결과 캐시는 쓰지 않습니다 (4.2.5에서 캐시 쓰기 중 오류가 났음)
)
print(f"채점 {len(test_cases)}문항 × 지표 {len(METRICS)}개, {time.time() - start:.0f}초\n")

by_id = {r["id"]: r for r in rows}
names = list(METRICS)
for r in rows:
    r["deepeval"] = {}
missing = []
for tr in result.test_results:
    if not tr.metrics_data or len(tr.metrics_data) < len(names):
        missing.append((tr.name, len(tr.metrics_data or [])))
    data = tr.metrics_data or []
    # 같은 이름("Faithfulness")의 지표가 둘이라, 지표를 넘긴 순서대로 이름을 붙입니다
    assert [md.name for md in data] == [m.__name__ for m in METRICS.values()][:len(data)]
    by_id[int(tr.name)]["deepeval"] = {name: md.score for name, md in zip(names, data)}
    by_id[int(tr.name)]["deepeval_reason"] = {name: md.reason for name, md in zip(names, data)}
no_result = sorted(r["id"] for r in rows if not r["deepeval"])
print(f"결과가 온 문항 {len(result.test_results)}개, 결과가 없는 문항 {no_result}, 지표가 빠진 문항 {missing}\n")

print(f"  {'지표':<22} 평균   0.7 미만 개수   RAGAS 대응 지표와 순위 상관")
for name in names:
    pairs = [(r["deepeval"][name], r["ragas"][RAGAS_PAIR[name]]) for r in rows
             if r["deepeval"].get(name) is not None and r["ragas"][RAGAS_PAIR[name]] is not None]
    values = [r["deepeval"][name] for r in rows if r["deepeval"].get(name) is not None]
    low = sorted(r["id"] for r in rows if r["deepeval"].get(name) is not None and r["deepeval"][name] < 0.7)
    rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
    print(f"  {name:<22} {sum(values) / len(values):.3f}  {len(low):2d} {low}  ρ={rho:.3f} ({RAGAS_PAIR[name]})")

out = Path("results") / "ch07_deepeval_c400_k3.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n결과 저장: {out}")
